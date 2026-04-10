import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

contact_points = []
image_display = None
base_image = None


def mouse_callback(event, x, y, flags, param):
    global contact_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(contact_points) == 2:
            return
        contact_points.append((x, y))


def refine_contour_subpixel(gray, contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)

    refined = cv2.cornerSubPix(
        gray, pts, (5, 5), (-1, -1), criteria
    )

    return refined.reshape(-1, 2)


def ellipse_line_intersections_float(p1, p2, cx, cy, a, b, theta):
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

    A = (A_val**2)/(a**2) + (B_val**2)/(b**2)
    B = 2*((X0p*A_val)/(a**2) + (Y0p*B_val)/(b**2))
    C = (X0p**2)/(a**2) + (Y0p**2)/(b**2) - 1

    if abs(A) < 1e-12:
        return []

    D = B**2 - 4*A*C
    if D < 0:
        return []

    sqrtD = math.sqrt(D)

    pts = []
    for t in [(-B + sqrtD)/(2*A), (-B - sqrtD)/(2*A)]:
        xi = x1 + t*dx
        yi = y1 + t*dy
        pts.append((xi, yi))

    return pts


def local_tangent_fit(contour_pts, contact_pt, radius=10):
    pts = np.array(contour_pts)
    dists = np.linalg.norm(pts - contact_pt, axis=1)
    local = pts[dists < radius]

    if len(local) < 5:
        return None

    mean = np.mean(local, axis=0)
    cov = np.cov((local - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)

    direction = eigvecs[:, np.argmax(eigvals)]
    angle = math.degrees(math.atan2(direction[1], direction[0]))

    return angle


def compute_contact_angle(tangent_angle, line_angle):
    diff = abs(tangent_angle - line_angle) % 180
    return 180 - diff


def select_image_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def main():
    global contact_points, image_display, base_image

    path = select_image_file()
    if not path:
        return

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    droplet = max(contours, key=cv2.contourArea)

    refined = refine_contour_subpixel(gray, droplet)

    ellipse = cv2.fitEllipse(refined.astype(np.float32))
    (cx, cy), (major, minor), angle_deg = ellipse

    a = major / 2
    b = minor / 2
    theta = math.radians(angle_deg)

    base_image = image.copy()
    cv2.ellipse(base_image, ellipse, (255, 0, 0), 2)

    cv2.namedWindow("Contact Angle")
    cv2.setMouseCallback("Contact Angle", mouse_callback)

    while True:
        if cv2.getWindowProperty("Contact Angle", cv2.WND_PROP_VISIBLE) < 1:
            break

        image_display = base_image.copy()

        for pt in contact_points:
            cv2.circle(image_display, pt, 5, (0, 255, 0), -1)

        # 状態表示
        if len(contact_points) == 0:
            status = "Now: select 2 points"
        elif len(contact_points) == 1:
            status = "Select second point"
        else:
            status = "Computed"

        cv2.putText(image_display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if len(contact_points) == 2:
            p1, p2 = contact_points
            cv2.line(image_display, p1, p2, (0, 255, 0), 2)

            line_angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

            inters = ellipse_line_intersections_float(p1, p2, cx, cy, a, b, theta)

            if len(inters) == 2:
                # 左右に分ける
                inters = sorted(inters, key=lambda p: p[0])

                angles = []

                for i, cp in enumerate(inters):
                    tangent_angle = local_tangent_fit(refined, cp)

                    if tangent_angle is not None:
                        angle = compute_contact_angle(tangent_angle, line_angle)
                        angles.append(angle)

                        cp_i = (int(cp[0]), int(cp[1]))
                        cv2.circle(image_display, cp_i, 5, (0, 0, 255), -1)

                        L = 100
                        pt1 = (int(cp_i[0] - L*np.cos(np.radians(tangent_angle))),
                               int(cp_i[1] - L*np.sin(np.radians(tangent_angle))))
                        pt2 = (int(cp_i[0] + L*np.cos(np.radians(tangent_angle))),
                               int(cp_i[1] + L*np.sin(np.radians(tangent_angle))))

                        cv2.line(image_display, pt1, pt2, (0, 255, 255), 2)

                if len(angles) == 2:
                    text = f"L:{angles[0]:.2f}  R:{angles[1]:.2f}"
                    cv2.putText(image_display, text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 操作表示
        cv2.putText(image_display, "r: reset  q: quit",
                    (10, image_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Contact Angle", image_display)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            contact_points = []

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
