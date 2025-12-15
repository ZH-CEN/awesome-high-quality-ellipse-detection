"""
Ellipse Detection Module - Main detector class

This module implements a high-quality ellipse detection algorithm based on
contour analysis and ellipse fitting. Uses OpenCV for robust detection.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Constants
GRADIENT_NOTDEF = -1024.0  # Value for undefined gradient angles
EIGENVALUE_TOLERANCE = 1e-8  # Tolerance for eigenvalue comparison in ellipse fitting


@dataclass
class Ellipse:
    """Represents a detected ellipse with its parameters."""
    center_x: float  # x coordinate of center
    center_y: float  # y coordinate of center
    semi_major: float  # semi-major axis (a)
    semi_minor: float  # semi-minor axis (b)
    angle: float  # rotation angle in radians

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        """Return ellipse parameters as a tuple (x, y, a, b, phi)."""
        return (self.center_x, self.center_y, self.semi_major, self.semi_minor, self.angle)
    
    def to_opencv_format(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Convert to OpenCV ellipse format ((cx, cy), (width, height), angle_deg)."""
        return (
            (self.center_x, self.center_y),
            (self.semi_major * 2, self.semi_minor * 2),
            np.degrees(self.angle)
        )
    
    @classmethod
    def from_opencv(cls, opencv_ellipse) -> 'Ellipse':
        """Create Ellipse from OpenCV ellipse format."""
        (cx, cy), (w, h), angle_deg = opencv_ellipse
        a = max(w, h) / 2
        b = min(w, h) / 2
        angle_rad = np.radians(angle_deg)
        if w < h:
            angle_rad += np.pi / 2
        return cls(cx, cy, a, b, angle_rad)


class EllipseDetector:
    """
    High-quality ellipse detector based on arc-support line segments.
    
    Parameters
    ----------
    Tac : float
        Threshold of elliptic angular coverage (0-360 degrees).
        Higher value means more complete ellipse required. Default: 165.
    Tr : float
        Ratio threshold for support inliers (0-1).
        Higher value requires more sufficient support. Default: 0.6.
    specified_polarity : int
        1: detect only positive polarity ellipses
        -1: detect only negative polarity ellipses
        0: detect all ellipses. Default: 0.
    """
    
    def __init__(self, Tac: float = 165, Tr: float = 0.6, specified_polarity: int = 0):
        self.Tac = Tac
        self.Tr = Tr
        self.specified_polarity = specified_polarity
        self.distance_tolerance = 2.0
        self.normal_tolerance = np.pi / 9  # 20 degrees
    
    def detect(self, image: np.ndarray) -> Tuple[List[Ellipse], np.ndarray]:
        """
        Detect ellipses in the input image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color).
            
        Returns
        -------
        ellipses : List[Ellipse]
            List of detected ellipses.
        edge_image : np.ndarray
            Edge image used for detection.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        min_dim = min(height, width)
        
        # Get edge image
        edge = self._compute_edge(gray)
        
        # Find contours
        contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        candidates = []
        
        # Method 1: Fit ellipses to contours
        for contour in contours:
            if len(contour) < 5:  # Need at least 5 points for ellipse fitting
                continue
            
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by size
            if area < 100 or perimeter < 30:
                continue
            
            # Fit ellipse using OpenCV
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (w, h), angle = ellipse
                
                # Validate ellipse parameters
                a = max(w, h) / 2
                b = min(w, h) / 2
                
                if a < 5 or b < 5:  # Too small
                    continue
                if a > min_dim * 0.9:  # Too large
                    continue
                if b / a < 0.1:  # Too elongated (nearly a line)
                    continue
                if cx < 0 or cx >= width or cy < 0 or cy >= height:
                    continue
                
                # Check ellipse quality using edge support
                support = self._compute_ellipse_support(ellipse, edge)
                
                # Angular coverage threshold
                min_coverage = self.Tac / 360.0
                if support['coverage'] >= min_coverage and support['ratio'] >= self.Tr:
                    angle_rad = np.radians(angle)
                    if w < h:
                        angle_rad += np.pi / 2
                    candidates.append(Ellipse(cx, cy, a, b, angle_rad))
                    
            except cv2.error:
                continue
        
        # Method 2: Try arc-based detection for partial ellipses
        arc_candidates = self._detect_from_arcs(gray, edge)
        candidates.extend(arc_candidates)
        
        # Remove duplicates
        ellipses = self._remove_duplicates(candidates)
        
        return ellipses, edge
    
    def _compute_edge(self, gray: np.ndarray) -> np.ndarray:
        """Compute edge image using adaptive Canny."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Compute adaptive thresholds based on median
        median = np.median(blurred)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        
        # Apply Canny edge detection
        edge = cv2.Canny(blurred, lower, upper)
        
        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = cv2.erode(edge, kernel, iterations=1)
        
        return edge
    
    def _compute_ellipse_support(self, ellipse, edge: np.ndarray) -> dict:
        """
        Compute edge support for an ellipse.
        
        Returns dict with 'ratio' (support ratio) and 'coverage' (angular coverage).
        """
        (cx, cy), (w, h), angle = ellipse
        height, width = edge.shape
        
        a = max(w, h) / 2
        b = min(w, h) / 2
        angle_rad = np.radians(angle)
        
        # Sample points on the ellipse
        n_samples = max(72, int(2 * np.pi * a))  # At least 72 samples
        t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Ellipse points
        x_local = a * np.cos(t)
        y_local = b * np.sin(t)
        
        # Rotate and translate
        x = cx + x_local * cos_a - y_local * sin_a
        y = cy + x_local * sin_a + y_local * cos_a
        
        # Count support pixels and compute coverage
        support_count = 0
        supported_angles = []
        
        tolerance = max(2, min(a, b) * 0.05)  # Adaptive tolerance
        
        for i in range(n_samples):
            xi, yi = int(round(x[i])), int(round(y[i]))
            
            # Check neighborhood for edge pixels
            found = False
            for dx in range(-int(tolerance), int(tolerance) + 1):
                for dy in range(-int(tolerance), int(tolerance) + 1):
                    px, py = xi + dx, yi + dy
                    if 0 <= px < width and 0 <= py < height:
                        if edge[py, px] > 0:
                            found = True
                            break
                if found:
                    break
            
            if found:
                support_count += 1
                supported_angles.append(t[i])
        
        ratio = support_count / n_samples if n_samples > 0 else 0
        
        # Compute angular coverage
        coverage = 0.0
        if len(supported_angles) > 1:
            supported_angles = np.array(supported_angles)
            supported_angles.sort()
            
            # Find the largest gap
            gaps = np.diff(supported_angles)
            gaps = np.append(gaps, 2 * np.pi - (supported_angles[-1] - supported_angles[0]))
            largest_gap = np.max(gaps)
            coverage = (2 * np.pi - largest_gap) / (2 * np.pi)
        
        return {'ratio': ratio, 'coverage': coverage}
    
    def _detect_from_arcs(self, gray: np.ndarray, edge: np.ndarray) -> List[Ellipse]:
        """Detect ellipses from arc segments using LSD."""
        candidates = []
        
        # Use LSD to detect line segments (arcs appear as connected segments)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)
        
        if lines is None or len(lines) < 3:
            return candidates
        
        # Group nearby line segments
        lines = lines.reshape(-1, 4)
        groups = self._group_segments(lines)
        
        # Try to fit ellipses to each group
        for group_indices in groups:
            if len(group_indices) < 3:
                continue
            
            # Collect all points from the group
            points = []
            for idx in group_indices:
                x1, y1, x2, y2 = lines[idx]
                points.append([x1, y1])
                points.append([x2, y2])
                # Add intermediate points
                for t in np.linspace(0, 1, 5):
                    points.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
            
            points = np.array(points, dtype=np.float32)
            
            if len(points) < 5:
                continue
            
            try:
                ellipse = cv2.fitEllipse(points)
                (cx, cy), (w, h), angle = ellipse
                
                a = max(w, h) / 2
                b = min(w, h) / 2
                
                height, width = gray.shape
                
                # Validate
                if a < 10 or b < 10:
                    continue
                if a > min(width, height) * 0.9:
                    continue
                if b / a < 0.1:
                    continue
                if cx < 0 or cx >= width or cy < 0 or cy >= height:
                    continue
                
                # Check support
                support = self._compute_ellipse_support(ellipse, edge)
                
                min_coverage = self.Tac / 360.0 * 0.5  # Relaxed for arc-based
                if support['coverage'] >= min_coverage and support['ratio'] >= self.Tr * 0.8:
                    angle_rad = np.radians(angle)
                    if w < h:
                        angle_rad += np.pi / 2
                    candidates.append(Ellipse(cx, cy, a, b, angle_rad))
                    
            except cv2.error:
                continue
        
        return candidates
    
    def _group_segments(self, lines: np.ndarray, 
                       dist_threshold: float = 15.0,
                       angle_threshold: float = 30.0) -> List[List[int]]:
        """Group line segments based on spatial and angular proximity."""
        n = len(lines)
        if n == 0:
            return []
        
        # Compute segment properties
        angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
        centers = np.column_stack([
            (lines[:, 0] + lines[:, 2]) / 2,
            (lines[:, 1] + lines[:, 3]) / 2
        ])
        
        # Build adjacency based on proximity and angle
        angle_threshold_rad = np.radians(angle_threshold)
        
        groups = []
        used = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if used[i]:
                continue
            
            group = [i]
            used[i] = True
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                
                for j in range(n):
                    if used[j]:
                        continue
                    
                    # Check angle difference
                    angle_diff = abs(angles[current] - angles[j])
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    if angle_diff > angle_threshold_rad:
                        continue
                    
                    # Check distance (endpoint to endpoint)
                    endpoints_i = lines[current].reshape(2, 2)
                    endpoints_j = lines[j].reshape(2, 2)
                    
                    min_dist = float('inf')
                    for pi in endpoints_i:
                        for pj in endpoints_j:
                            d = np.sqrt(np.sum((pi - pj)**2))
                            min_dist = min(min_dist, d)
                    
                    if min_dist < dist_threshold:
                        group.append(j)
                        used[j] = True
                        queue.append(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _compute_edge_and_gradient(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute edge image and gradient angles using Canny edge detection."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Compute gradients
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angles = np.arctan2(gx, -gy)  # Level line angle (perpendicular to gradient)
        
        # Apply Canny edge detection
        edge = cv2.Canny(blurred, 50, 150)
        
        # Mask angles where there's no edge
        angles[edge == 0] = GRADIENT_NOTDEF
        
        return edge, angles
    
    def _detect_line_segments(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect line segments using LSD (Line Segment Detector)."""
        # Create LSD detector
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # Detect lines
        lines, widths, precs, nfas = lsd.detect(gray)
        
        if lines is None:
            return None
        
        # Reshape and add additional info
        n_lines = len(lines)
        result = np.zeros((n_lines, 8), dtype=np.float64)
        
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            
            # Compute direction and length
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx /= length
                dy /= length
            
            # Store: x1, y1, x2, y2, dx, dy, length, polarity
            result[i] = [x1, y1, x2, y2, dx, dy, length, 1]
        
        # Filter very short lines
        result = result[result[:, 6] >= 5]
        
        return result
    
    def _group_line_segments(self, lines: np.ndarray, 
                             img_shape: Tuple[int, int]) -> Tuple[List[List[int]], np.ndarray]:
        """
        Group line segments into arcs based on spatial and angular proximity.
        
        Returns groups of line segment indices and their angular coverages.
        """
        n_lines = len(lines)
        if n_lines == 0:
            return [], np.array([])
        
        # Compute angles for each line segment
        angles = np.arctan2(lines[:, 5], lines[:, 4])  # atan2(dy, dx)
        
        # Sort by angle for grouping
        sorted_indices = np.argsort(angles)
        
        groups = []
        coverages = []
        used = np.zeros(n_lines, dtype=bool)
        
        distance_threshold = 10.0  # pixels
        angle_threshold = np.pi / 9  # 20 degrees
        
        for start_idx in range(n_lines):
            if used[start_idx]:
                continue
            
            current_group = [start_idx]
            used[start_idx] = True
            
            # Try to extend the group
            current_angle = angles[start_idx]
            current_endpoint = lines[start_idx, 2:4]  # (x2, y2)
            
            for next_idx in range(n_lines):
                if used[next_idx]:
                    continue
                
                # Check angle compatibility
                angle_diff = abs(angles[next_idx] - current_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                if angle_diff > angle_threshold:
                    continue
                
                # Check spatial proximity (endpoint to startpoint)
                next_startpoint = lines[next_idx, 0:2]  # (x1, y1)
                dist = np.sqrt(np.sum((current_endpoint - next_startpoint)**2))
                
                if dist < distance_threshold:
                    current_group.append(next_idx)
                    used[next_idx] = True
                    current_endpoint = lines[next_idx, 2:4]
                    current_angle = angles[next_idx]
            
            if len(current_group) >= 1:
                groups.append(current_group)
                
                # Compute angular coverage of the group
                group_angles = angles[current_group]
                coverage = self._compute_coverage(lines[current_group])
                coverages.append(coverage)
        
        return groups, np.array(coverages)
    
    def _compute_coverage(self, group_lines: np.ndarray) -> float:
        """Compute the angular coverage of a group of line segments."""
        if len(group_lines) == 0:
            return 0.0
        
        # Compute centroid
        all_points = np.vstack([
            group_lines[:, 0:2],  # start points
            group_lines[:, 2:4]   # end points
        ])
        centroid = np.mean(all_points, axis=0)
        
        # Compute angles from centroid
        vectors = all_points - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Compute coverage as max angle difference
        if len(angles) < 2:
            return 0.0
        
        angles_sorted = np.sort(angles)
        gaps = np.diff(angles_sorted)
        gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
        
        coverage = 2 * np.pi - np.max(gaps)
        return np.degrees(coverage)
    
    def _generate_candidates(self, lines: np.ndarray, groups: List[List[int]],
                            coverages: np.ndarray, angles: np.ndarray) -> List[Ellipse]:
        """Generate ellipse candidates from grouped line segments."""
        candidates = []
        
        # Get candidates from single high-coverage groups
        for i, group in enumerate(groups):
            if coverages[i] >= 80:  # 80 degrees minimum coverage
                if len(group) >= 3:
                    points = self._get_group_points(lines, group)
                    ellipse = self._fit_ellipse(points)
                    if ellipse is not None:
                        candidates.append(ellipse)
        
        # Get candidates from pairs of groups
        n_groups = len(groups)
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                # Check if groups can form an ellipse
                if coverages[i] + coverages[j] >= self.Tac:
                    points = np.vstack([
                        self._get_group_points(lines, groups[i]),
                        self._get_group_points(lines, groups[j])
                    ])
                    if len(points) >= 5:
                        ellipse = self._fit_ellipse(points)
                        if ellipse is not None:
                            candidates.append(ellipse)
        
        return candidates
    
    def _get_group_points(self, lines: np.ndarray, group: List[int]) -> np.ndarray:
        """Get all endpoints of lines in a group."""
        points = []
        for idx in group:
            points.append([lines[idx, 0], lines[idx, 1]])  # start point
            points.append([lines[idx, 2], lines[idx, 3]])  # end point
        return np.array(points)
    
    def _fit_ellipse(self, points: np.ndarray) -> Optional[Ellipse]:
        """
        Fit an ellipse to a set of points using least squares.
        
        Uses the direct least squares fitting method.
        """
        if len(points) < 5:
            return None
        
        points = points.astype(np.float64)
        
        # Build design matrix D
        x = points[:, 0]
        y = points[:, 1]
        
        D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
        
        # Build scatter matrix S
        S = D.T @ D
        
        # Build constraint matrix C
        C = np.zeros((6, 6))
        C[0, 2] = 2
        C[1, 1] = -1
        C[2, 0] = 2
        
        try:
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S) @ C)
            
            # Find the positive eigenvalue (should be close to 0 from positive side)
            valid_idx = np.where((np.real(eigenvalues) > -EIGENVALUE_TOLERANCE) & 
                                 (np.isfinite(eigenvalues)) & 
                                 (np.abs(np.imag(eigenvalues)) < EIGENVALUE_TOLERANCE))[0]
            
            if len(valid_idx) == 0:
                return None
            
            # Get the eigenvector for the smallest positive eigenvalue
            idx = valid_idx[np.argmin(np.abs(np.real(eigenvalues[valid_idx])))]
            A = np.real(eigenvectors[:, idx])
            
            # Ensure A[0] > 0 (coefficient of x^2)
            if A[0] < 0:
                A = -A
            
            # Convert to ellipse parameters
            ellipse = self._coefficients_to_ellipse(A)
            return ellipse
            
        except (np.linalg.LinAlgError, ValueError):
            return None
    
    def _coefficients_to_ellipse(self, A: np.ndarray) -> Optional[Ellipse]:
        """
        Convert ellipse coefficients (Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0)
        to geometric parameters (center_x, center_y, semi_major, semi_minor, angle).
        """
        a, b, c, d, e, f = A
        
        # Compute rotation angle
        theta = 0.5 * np.arctan2(b, a - c)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sin2 = sin_t ** 2
        cos2 = cos_t ** 2
        
        # Transform to canonical form
        Ao = f
        Au = d * cos_t + e * sin_t
        Av = -d * sin_t + e * cos_t
        Auu = a * cos2 + c * sin2 + b * cos_t * sin_t
        Avv = a * sin2 + c * cos2 - b * cos_t * sin_t
        
        # Compute center in rotated coordinates
        if abs(Auu) < 1e-10 or abs(Avv) < 1e-10:
            return None
        
        tu_center = -Au / (2 * Auu)
        tv_center = -Av / (2 * Avv)
        
        w_center = Ao - Auu * tu_center**2 - Avv * tv_center**2
        
        # Transform center back to original coordinates
        u_center = tu_center * cos_t - tv_center * sin_t
        v_center = tu_center * sin_t + tv_center * cos_t
        
        # Compute semi-axes
        Ru = -w_center / Auu
        Rv = -w_center / Avv
        
        if Ru <= 0 or Rv <= 0:
            return None
        
        Ru = np.sqrt(Ru)
        Rv = np.sqrt(Rv)
        
        # Ensure semi_major > semi_minor
        if Ru < Rv:
            Ru, Rv = Rv, Ru
            if theta < 0:
                theta += np.pi / 2
            else:
                theta -= np.pi / 2
        
        # Validate ellipse
        if Ru < 3 or Rv < 3:  # Minimum size
            return None
        
        if Ru / Rv > 20:  # Maximum eccentricity
            return None
        
        return Ellipse(u_center, v_center, Ru, Rv, theta)
    
    def _validate_candidates(self, candidates: List[Ellipse], 
                            edge: np.ndarray, angles: np.ndarray) -> List[Ellipse]:
        """Validate ellipse candidates using edge support."""
        validated = []
        height, width = edge.shape
        
        for candidate in candidates:
            # Check if center is within image bounds
            if not (0 <= candidate.center_x < width and 0 <= candidate.center_y < height):
                continue
            
            # Check if semi-axes are reasonable
            max_dim = max(width, height)
            if candidate.semi_major > max_dim or candidate.semi_minor > max_dim:
                continue
            
            # Compute edge support ratio
            support_ratio = self._compute_support_ratio(candidate, edge, angles)
            
            if support_ratio >= self.Tr:
                validated.append(candidate)
        
        # Remove duplicate ellipses
        validated = self._remove_duplicates(validated)
        
        return validated
    
    def _compute_support_ratio(self, ellipse: Ellipse, edge: np.ndarray, 
                               angles: np.ndarray) -> float:
        """Compute the ratio of edge pixels supporting the ellipse."""
        height, width = edge.shape
        
        # Sample points on the ellipse
        n_samples = int(2 * np.pi * max(ellipse.semi_major, ellipse.semi_minor))
        n_samples = max(36, min(360, n_samples))
        
        t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        
        # Compute ellipse points
        cos_phi = np.cos(ellipse.angle)
        sin_phi = np.sin(ellipse.angle)
        
        x_ellipse = ellipse.semi_major * np.cos(t)
        y_ellipse = ellipse.semi_minor * np.sin(t)
        
        # Rotate and translate
        x = ellipse.center_x + x_ellipse * cos_phi - y_ellipse * sin_phi
        y = ellipse.center_y + x_ellipse * sin_phi + y_ellipse * cos_phi
        
        # Count support pixels
        support_count = 0
        total_count = 0
        
        for i in range(n_samples):
            xi, yi = int(round(x[i])), int(round(y[i]))
            
            # Check neighborhood for edge pixels
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = xi + dx, yi + dy
                    if 0 <= px < width and 0 <= py < height:
                        if edge[py, px] > 0:
                            support_count += 1
                            break
                else:
                    continue
                break
            
            total_count += 1
        
        return support_count / total_count if total_count > 0 else 0
    
    def _remove_duplicates(self, ellipses: List[Ellipse], 
                          center_thresh: float = 5.0,
                          axis_thresh: float = 0.1) -> List[Ellipse]:
        """Remove duplicate ellipses based on similarity."""
        if len(ellipses) <= 1:
            return ellipses
        
        unique = []
        
        for ellipse in ellipses:
            is_duplicate = False
            
            for existing in unique:
                # Check center distance
                center_dist = np.sqrt(
                    (ellipse.center_x - existing.center_x)**2 + 
                    (ellipse.center_y - existing.center_y)**2
                )
                
                if center_dist > center_thresh:
                    continue
                
                # Check axis similarity
                a_diff = abs(ellipse.semi_major - existing.semi_major) / max(ellipse.semi_major, existing.semi_major)
                b_diff = abs(ellipse.semi_minor - existing.semi_minor) / max(ellipse.semi_minor, existing.semi_minor)
                
                if a_diff < axis_thresh and b_diff < axis_thresh:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(ellipse)
        
        return unique
    
    def draw_ellipses(self, image: np.ndarray, ellipses: List[Ellipse],
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """Draw detected ellipses on the image."""
        result = image.copy()
        
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for ellipse in ellipses:
            center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
            axes = (int(round(ellipse.semi_major)), int(round(ellipse.semi_minor)))
            angle = int(round(np.degrees(ellipse.angle)))
            
            cv2.ellipse(result, center, axes, angle, 0, 360, color, thickness)
            
            # Draw center point
            cv2.circle(result, center, 3, (0, 0, 255), -1)
        
        return result


def detect_ellipses(image: np.ndarray, Tac: float = 165, Tr: float = 0.6,
                    specified_polarity: int = 0) -> Tuple[List[Ellipse], np.ndarray]:
    """
    Convenience function to detect ellipses in an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    Tac : float
        Angular coverage threshold (default: 165 degrees).
    Tr : float
        Support inliers ratio threshold (default: 0.6).
    specified_polarity : int
        Polarity filter: 1 (positive), -1 (negative), 0 (all).
        
    Returns
    -------
    ellipses : List[Ellipse]
        List of detected ellipses.
    edge_image : np.ndarray
        Edge image used for detection.
    """
    detector = EllipseDetector(Tac=Tac, Tr=Tr, specified_polarity=specified_polarity)
    return detector.detect(image)
