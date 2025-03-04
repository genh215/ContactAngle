import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

contact_points = []
image_display = None
base_image = None

def mouse_callback(event, x, y, flags, param):
    global contact_points, image_display, base_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(contact_points) == 2:
            contact_points = []
            image_display = base_image.copy()

        contact_points.append((x, y))
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)

        if len(contact_points) == 2:
            cv2.line(image_display, contact_points[0], contact_points[1], (0, 255, 0), 2)

        cv2.imshow("Contact Angle", image_display)

def ellipse_line_intersections(p1, p2, cx, cy, a, b, theta):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    X0 = x1 - cx
    Y0 = y1 - cy
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    X0p = X0 * cos_t + Y0 * sin_t
    Y0p = -X0 * sin_t + Y0 * cos_t
    A_val = dx * cos_t + dy * sin_t
    B_val = -dx * sin_t + dy * cos_t
    A_coef = (A_val**2) / (a**2) + (B_val**2) / (b**2)
    B_coef = 2 * ((X0p * A_val) / (a**2) + (Y0p * B_val) / (b**2))
    C_coef = (X0p**2) / (a**2) + (Y0p**2) / (b**2) - 1
    discriminant = B_coef**2 - 4 * A_coef * C_coef
    intersections = []

    if discriminant < 0:
        return intersections
    
    sqrt_disc = math.sqrt(discriminant)

    for t in [(-B_coef + sqrt_disc) / (2 * A_coef), (-B_coef - sqrt_disc) / (2 * A_coef)]:
        xi = x1 + t * dx
        yi = y1 + t * dy
        intersections.append((int(round(xi)), int(round(yi))))

    return intersections

def compute_ellipse_interior_tangent_angle(point, cx, cy, a, b, theta):
    x, y = point
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    X = x - cx
    Y = y - cy
    dx_prime = X * cos_t + Y * sin_t
    dy_prime = -X * sin_t + Y * cos_t
    dF_dx = 2 * (dx_prime / (a**2)) * cos_t - 2 * (dy_prime / (b**2)) * sin_t
    dF_dy = 2 * (dx_prime / (a**2)) * sin_t + 2 * (dy_prime / (b**2)) * cos_t
    normal = (dF_dx, dF_dy)
    v_center = (cx - x, cy - y)

    if normal[0] * v_center[0] + normal[1] * v_center[1] < 0:
        normal = (-normal[0], -normal[1])

    tangent = (-normal[1], normal[0])
    tangent_angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    return tangent_angle

def compute_contact_angle(tangent_angle, contact_line_angle):
    diff = abs(tangent_angle - contact_line_angle) % 180
    contact_angle = 180 - diff
    return contact_angle

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("TIFF/JPEG files", "*.tif *.tiff *.jpeg *.jpg")]
    )
    return file_path

def main():
    global image_display, contact_points, base_image
    image_path = select_image_file()

    if not image_path:
        print("No image file selected.")
        return
    
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours detected.")
        return
    
    droplet_contour = max(contours, key=cv2.contourArea)

    if len(droplet_contour) < 5:
        print("Not enough contour points.")
        return
    
    ellipse = cv2.fitEllipse(droplet_contour)
    (cx, cy), (major, minor), angle_deg = ellipse
    a = major / 2.0
    b = minor / 2.0
    theta = math.radians(angle_deg)
    image_display = image.copy()

    cv2.ellipse(image_display, ellipse, (255, 0, 0), 2)
    base_image = image_display.copy()
    cv2.namedWindow("Contact Angle")
    cv2.setMouseCallback("Contact Angle", mouse_callback)

    while True:
        cv2.imshow("Contact Angle", image_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or cv2.getWindowProperty("Contact Angle", cv2.WND_PROP_VISIBLE) < 1:
            break

        if len(contact_points) == 2:
            (x1, y1), (x2, y2) = contact_points
            contact_line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            intersections = ellipse_line_intersections(contact_points[0], contact_points[1],
                                                       cx, cy, a, b, theta)
            if intersections:
                contact_pt = max(intersections, key=lambda p: p[1])
                cv2.circle(image_display, contact_pt, 5, (0, 0, 255), -1)
                tangent_angle = compute_ellipse_interior_tangent_angle(contact_pt, cx, cy, a, b, theta)
                contact_angle = compute_contact_angle(tangent_angle, contact_line_angle)
                tangent_length = 100
                pt1 = (int(contact_pt[0] - tangent_length * math.cos(math.radians(tangent_angle))),
                       int(contact_pt[1] - tangent_length * math.sin(math.radians(tangent_angle))))
                pt2 = (int(contact_pt[0] + tangent_length * math.cos(math.radians(tangent_angle))),
                       int(contact_pt[1] + tangent_length * math.sin(math.radians(tangent_angle))))
                cv2.line(image_display, pt1, pt2, (0, 255, 255), 2)
                result_text = f"{contact_angle:.1f} deg"
                cv2.putText(image_display, result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image_display, "Error", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
