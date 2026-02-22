"""
Mask R-CNN Demo Script
======================
This script demonstrates how to use PyTorch's torchvision Mask R-CNN model
for instance segmentation. The pretrained model can detect 91 COCO classes
including animals, vehicles, and common objects.

Usage:
    python demo_mask_rcnn.py                    # Use default sample image
    python demo_mask_rcnn.py path/to/image.jpg  # Use your own image
"""

import sys
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# COCO class names (91 classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colors for visualization (one per class, cycling)
COLORS = plt.cm.hsv(np.linspace(0, 1, len(COCO_CLASSES))).tolist()


def load_model(device='auto'):
    """
    Load the pretrained Mask R-CNN model.
    
    Args:
        device: 'cuda', 'cpu', or 'auto' (auto-detect)
    
    Returns:
        model: The loaded model
        device: The device being used
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Mask R-CNN model on {device}...")
    
    # Load pretrained model with COCO weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, device


def load_image(image_path):
    """
    Load and preprocess an image for the model.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        image_tensor: Tensor ready for model input
        original_image: PIL Image for visualization
    """
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image)
    return image_tensor, image


def run_inference(model, image_tensor, device, confidence_threshold=0.5):
    """
    Run Mask R-CNN inference on an image.
    
    Args:
        model: The Mask R-CNN model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        confidence_threshold: Minimum confidence score to keep detections
    
    Returns:
        predictions: Filtered predictions dictionary
    """
    print("Running inference...")
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])[0]
    
    # Filter by confidence threshold
    keep = predictions['scores'] > confidence_threshold
    
    filtered_predictions = {
        'boxes': predictions['boxes'][keep].cpu(),
        'labels': predictions['labels'][keep].cpu(),
        'scores': predictions['scores'][keep].cpu(),
        'masks': predictions['masks'][keep].cpu()
    }
    
    num_detections = len(filtered_predictions['boxes'])
    print(f"Found {num_detections} object(s) with confidence > {confidence_threshold}")
    
    return filtered_predictions


def visualize_results(image, predictions, save_path='output.png', show=True):
    """
    Visualize detection results with bounding boxes and masks.
    
    Args:
        image: Original PIL image
        predictions: Model predictions dictionary
        save_path: Path to save the output image
        show: Whether to display the image
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original image with bounding boxes
    axes[0].imshow(image)
    axes[0].set_title('Detections with Bounding Boxes', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Image with segmentation masks
    axes[1].imshow(image)
    axes[1].set_title('Instance Segmentation Masks', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Draw detections
    for i in range(len(predictions['boxes'])):
        box = predictions['boxes'][i].numpy()
        label_idx = predictions['labels'][i].item()
        score = predictions['scores'][i].item()
        mask = predictions['masks'][i, 0].numpy()
        
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f'class_{label_idx}'
        color = COLORS[label_idx % len(COLORS)]
        
        # Draw bounding box on left image
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[0].add_patch(rect)
        
        # Add label text
        axes[0].text(
            x1, y1 - 5,
            f'{label_name}: {score:.2f}',
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
        )
        
        # Draw mask on right image
        masked = np.where(mask > 0.5)
        color_mask = np.zeros((*mask.shape, 4))
        color_mask[masked[0], masked[1]] = [*color[:3], 0.5]  # RGBA with alpha
        axes[1].imshow(color_mask)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_detections(predictions):
    """Print a summary of all detections."""
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    
    for i in range(len(predictions['boxes'])):
        label_idx = predictions['labels'][i].item()
        score = predictions['scores'][i].item()
        box = predictions['boxes'][i].numpy()
        
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f'class_{label_idx}'
        
        print(f"\n[{i+1}] {label_name}")
        print(f"    Confidence: {score:.2%}")
        print(f"    Bounding Box: ({box[0]:.0f}, {box[1]:.0f}) to ({box[2]:.0f}, {box[3]:.0f})")
    
    print("\n" + "="*50)


def main():
    """Main function to run the demo."""
    print("\n" + "="*60)
    print("  MASK R-CNN INSTANCE SEGMENTATION DEMO")
    print("  Using PyTorch torchvision pretrained model")
    print("="*60 + "\n")
    
    # Get image path from command line or use sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("No image provided. Attempting to download sample manatee image...")
        image_path = create_sample_image()
        if image_path is None:
            return
    
    # Load model
    model, device = load_model()
    
    # Load and process image
    image_tensor, original_image = load_image(image_path)
    
    # Run inference
    predictions = run_inference(model, image_tensor, device, confidence_threshold=0.5)
    
    # Print results
    print_detections(predictions)
    
    # Visualize
    visualize_results(original_image, predictions, save_path='detection_output.png', show=True)


if __name__ == '__main__':
    main()


# See PyTorch tutorial for full details:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# =============================================================================

