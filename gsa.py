import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2

# Add paths to access required modules
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))
sys.path.append(os.path.join(os.path.dirname(__file__), "segment_anything"))

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor


class GroundedSegmentAnything:
    """
    Grounded Segment Anything - combines Grounding DINO and SAM models
    to generate masks from text prompts.
    """
    
    def __init__(
        self,
        groundingdino_config,
        groundingdino_model,
        sam_vit_model,
        device="cuda",
        sam_version="vit_h",
        box_threshold=0.3,
        text_threshold=0.25,
        use_sam_hq=False,
        sam_hq_checkpoint=None,
        bert_base_uncased_path=None,
    ):
        """
        Initialize GSA with required models.
        
        Args:
            groundingdino_config: Path to the Grounding DINO config file
            groundingdino_model: Path to the Grounding DINO checkpoint
            sam_vit_model: Path to the SAM checkpoint
            device: Device to run models on ("cuda" or "cpu")
            sam_version: SAM model version ("vit_b", "vit_l", or "vit_h")
            box_threshold: Threshold for box detection
            text_threshold: Threshold for text detection
            use_sam_hq: Whether to use SAM-HQ
            sam_hq_checkpoint: Path to SAM-HQ checkpoint (if use_sam_hq is True)
            bert_base_uncased_path: Path to bert_base_uncased model (optional)
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Initialize Grounding DINO model
        args = SLConfig.fromfile(groundingdino_config)
        args.device = device
        if bert_base_uncased_path:
            args.bert_base_uncased_path = bert_base_uncased_path
        self.grounding_model = build_model(args)
        checkpoint = torch.load(groundingdino_model, map_location="cpu")
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model.eval()
        self.grounding_model.to(device)
        
        # Initialize SAM model
        if use_sam_hq:
            self.sam_predictor = SamPredictor(
                sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
            )
        else:
            self.sam_predictor = SamPredictor(
                sam_model_registry[sam_version](checkpoint=sam_vit_model).to(device)
            )
        
        # Image transformation for Grounding DINO
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def _load_image(self, image):
        """
        Load and transform image for Grounding DINO.
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (original PIL image, transformed tensor image)
        """
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type")
            
        image_tensor, _ = self.transform(image_pil, None)
        return image_pil, image_tensor
    
    def _get_grounding_output(self, image_tensor, caption):
        """
        Run the Grounding DINO model to get bounding boxes.
        
        Args:
            image_tensor: Transformed image tensor
            caption: Text prompt
            
        Returns:
            tuple: (filtered boxes, predicted phrases)
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
            
        with torch.no_grad():
            outputs = self.grounding_model(image_tensor.unsqueeze(0).to(self.device), 
                                         captions=[caption])
                
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # Get phrases
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, 
                                                 tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            
        return boxes_filt, pred_phrases
    
    def process(self, image, prompt):
        """
        Process an image with a text prompt to get a segmentation mask.
        
        Args:
            image: PIL Image, numpy array, or path to image
            prompt: Text prompt describing what to segment
            
        Returns:
            numpy.ndarray: Binary mask of the segmented object(s)
        """
        # Load and prepare image
        image_pil, image_tensor = self._load_image(image)
        
        # Convert PIL image to numpy for SAM
        if isinstance(image, str):
            image_np = cv2.imread(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image_pil)
            
        # Get bounding boxes from Grounding DINO
        boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, prompt)
        
        # No objects found
        if len(boxes_filt) == 0:
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
        
        # Prepare SAM
        self.sam_predictor.set_image(image_np)
        
        # Scale boxes to image size
        H, W = image_pil.height, image_pil.width
        scaled_boxes = []
        
        for box in boxes_filt:
            scaled_box = box.clone()
            scaled_box = scaled_box * torch.Tensor([W, H, W, H])
            scaled_box[:2] -= scaled_box[2:] / 2
            scaled_box[2:] += scaled_box[:2]
            scaled_boxes.append(scaled_box)
            
        scaled_boxes = torch.stack(scaled_boxes)
        
        # Transform boxes for SAM
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            scaled_boxes, image_np.shape[:2]
        ).to(self.device)
        
        # Get masks from SAM
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Combine all masks into a single mask
        combined_mask = torch.zeros(masks.shape[-2:], dtype=torch.bool, device=masks.device)
        for mask in masks:
            combined_mask |= mask[0]  # Combine with OR operation
            
        # Convert to numpy and return
        return combined_mask.cpu().numpy().astype(np.uint8)
        
    def __del__(self):
        """Clean up resources when the object is deleted"""
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except:
            pass


# Example usage (for testing)
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="GSA Demo")
    parser.add_argument("--config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config")
    parser.add_argument("--groundingdino_model", type=str, default="../../downloaded_models/gsa/groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint")
    parser.add_argument("--sam_vit_model", type=str, default="../../downloaded_models/gsa/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_path", type=str, default="output.png", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Initialize model
    gsa = GroundedSegmentAnything(
        groundingdino_config=args.config,
        groundingdino_model=args.groundingdino_model,
        sam_vit_model=args.sam_vit_model,
        device=args.device,
    )
    
    # Process image
    image = Image.open(args.input_image)
    mask = gsa.process(image, args.text_prompt)
    
    # Save output
    mask_img = Image.fromarray(mask * 255)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    mask_img.save(args.output_path)
    print(f"Mask saved to {args.output_path}")
