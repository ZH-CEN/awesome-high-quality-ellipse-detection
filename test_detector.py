#!/usr/bin/env python3
"""
Test script for the Python ellipse detector.

Run this script to verify the ellipse detection implementation works correctly.
"""

import os
import sys
import time
import tempfile

# Add python_src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_src'))

import cv2
import numpy as np
from ellipse_detector import EllipseDetector, detect_ellipses

# Get temp directory for cross-platform compatibility
TEMP_DIR = tempfile.gettempdir()


def test_synthetic_image():
    """Test with a synthetic image containing ellipses."""
    print("=" * 60)
    print("Test 1: Synthetic image with drawn ellipses")
    print("=" * 60)
    
    # Create a white image
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    # Draw some ellipses
    cv2.ellipse(img, (200, 200), (80, 50), 30, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(img, (500, 300), (100, 60), -15, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(img, (350, 400), (60, 60), 0, 0, 360, (0, 0, 0), 2)  # Circle
    
    # Detect ellipses
    detector = EllipseDetector(Tac=165, Tr=0.5)
    start_time = time.time()
    ellipses, edge = detector.detect(img)
    elapsed = time.time() - start_time
    
    print(f"Detected {len(ellipses)} ellipses in {elapsed*1000:.1f}ms")
    for i, e in enumerate(ellipses):
        print(f"  Ellipse {i+1}: center=({e.center_x:.1f}, {e.center_y:.1f}), "
              f"axes=({e.semi_major:.1f}, {e.semi_minor:.1f}), "
              f"angle={np.degrees(e.angle):.1f}°")
    
    # Draw results
    result = detector.draw_ellipses(img, ellipses)
    
    # Save result
    output_path = os.path.join(TEMP_DIR, 'test_synthetic_result.jpg')
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
    
    return len(ellipses) > 0


def test_real_image(image_path):
    """Test with a real image."""
    print("=" * 60)
    print(f"Test 2: Real image - {os.path.basename(image_path)}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Detect ellipses with lower thresholds for better detection
    detector = EllipseDetector(Tac=150, Tr=0.4)
    start_time = time.time()
    ellipses, edge = detector.detect(img)
    elapsed = time.time() - start_time
    
    print(f"Detected {len(ellipses)} ellipses in {elapsed*1000:.1f}ms")
    for i, e in enumerate(ellipses):
        print(f"  Ellipse {i+1}: center=({e.center_x:.1f}, {e.center_y:.1f}), "
              f"axes=({e.semi_major:.1f}, {e.semi_minor:.1f}), "
              f"angle={np.degrees(e.angle):.1f}°")
    
    # Draw results
    result = detector.draw_ellipses(img, ellipses)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(TEMP_DIR, f'test_{base_name}_result.jpg')
    edge_path = os.path.join(TEMP_DIR, f'test_{base_name}_edge.jpg')
    
    cv2.imwrite(output_path, result)
    cv2.imwrite(edge_path, edge)
    
    print(f"Result saved to: {output_path}")
    print(f"Edge image saved to: {edge_path}")
    
    return True


def test_api():
    """Test the convenience function API."""
    print("=" * 60)
    print("Test 3: Convenience function API")
    print("=" * 60)
    
    # Create simple test image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 200
    cv2.ellipse(img, (200, 150), (70, 40), 0, 0, 360, (50, 50, 50), 2)
    
    # Use convenience function
    ellipses, edge = detect_ellipses(img, Tac=150, Tr=0.5)
    
    print(f"API test passed: detected {len(ellipses)} ellipses")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  High-Quality Ellipse Detection - Python Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Synthetic image
    try:
        results.append(("Synthetic image", test_synthetic_image()))
    except Exception as e:
        print(f"Test failed with error: {e}")
        results.append(("Synthetic image", False))
    
    print()
    
    # Test 2: Real images from pics directory
    pics_dir = os.path.join(os.path.dirname(__file__), 'pics')
    test_images = ['666.jpg', '43.jpg', '23.bmp']
    
    for img_name in test_images:
        img_path = os.path.join(pics_dir, img_name)
        try:
            results.append((f"Real image ({img_name})", test_real_image(img_path)))
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append((f"Real image ({img_name})", False))
        print()
    
    # Test 3: API test
    try:
        results.append(("API function", test_api()))
    except Exception as e:
        print(f"Test failed with error: {e}")
        results.append(("API function", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print("=" * 60 + "\n")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
