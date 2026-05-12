import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────
ACCIDENT_THRESHOLD  = 0.75   # minimum confidence to count as accident
CONSECUTIVE_FRAMES  = 3      # how many frames in a row before confirming
# ─────────────────────────────────────────────────────────────────

model = AccidentDetectionModel("model.json", "model_weights.h5")
font  = cv2.FONT_HERSHEY_SIMPLEX


def get_severity(prob):
    """Return (label, BGR color) based on confidence level."""
    if prob >= 0.90:
        return "CRITICAL", (0, 0, 220)        # Red
    elif prob >= 0.75:
        return "MODERATE", (0, 140, 255)       # Orange
    else:
        return "MINOR",    (0, 215, 255)        # Yellow


def draw_overlay(frame, pred, prob, consecutive, confirmed, alert_count, frame_num):
    """Draw HUD overlay on the frame."""
    h, w = frame.shape[:2]

    if pred == "Accident" and prob >= ACCIDENT_THRESHOLD:
        severity, color = get_severity(prob)

        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)

        # Blinking red border when confirmed
        if confirmed:
            blink = (frame_num // 10) % 2 == 0
            border_color = (0, 0, 255) if blink else (0, 0, 180)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 4)

        # Accident label + confidence
        label = f"ACCIDENT  {prob:.1%}"
        cv2.putText(frame, label, (10, 30), font, 0.85, color, 2)

        # Severity tag
        sev_label = f"Severity: {severity}"
        cv2.putText(frame, sev_label, (10, 52), font, 0.55, color, 1)

        # Right side — frame counter info
        info1 = f"Consec: {consecutive}/{CONSECUTIVE_FRAMES}"
        info2 = f"Alerts: {alert_count}"
        cv2.putText(frame, info1, (w - 180, 25), font, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, info2, (w - 180, 50), font, 0.55, (200, 200, 200), 1)

        # Confirmed stamp
        if confirmed:
            stamp = "!! CONFIRMED !!"
            ts, _ = cv2.getTextSize(stamp, font, 0.75, 2)
            tx = (w - ts[0]) // 2
            cv2.putText(frame, stamp, (tx, h - 15), font, 0.75, (0, 0, 255), 2)

    else:
        # Normal / below threshold
        cv2.rectangle(frame, (0, 0), (200, 35), (0, 0, 0), -1)
        cv2.putText(frame, "No Accident", (10, 25), font, 0.7, (0, 220, 0), 2)

    # Always show frame number bottom-right
    cv2.putText(frame, f"Frame: {frame_num}", (w - 130, h - 10),
                font, 0.45, (180, 180, 180), 1)

    return frame


def startapplication():
    video_path = r'D:\Accident-Detection-System\final_test_video\accident-test_57uuabZF.mp4'

    if not os.path.exists(video_path):
        print("ERROR: Video file not found at:", video_path)
        return

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("ERROR: Could not open video.")
        return

    # ── State tracking ────────────────────────────────────────────
    consecutive   = 0        # current streak of accident frames
    confirmed     = False    # has accident been confirmed (streak met)?
    confirm_frame = None     # frame number when first confirmed
    alerts        = []       # list of {frame, prob}
    all_probs     = []       # probability every frame (for post-run plot)
    frame_num     = 0
    # ─────────────────────────────────────────────────────────────

    print("\n▶  Starting Accident Detection — press Q to quit\n")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Video ended.")
            break

        frame_num += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi       = cv2.resize(rgb_frame, (250, 250))
        pred, prob_arr = model.predict_accident(roi[np.newaxis, :, :])

        # prob[0][0] = probability of class 0 = Accident
        prob = float(prob_arr[0][0])
        all_probs.append(prob)

        # ── Consecutive confirmation logic ────────────────────────
        if pred == "Accident" and prob >= ACCIDENT_THRESHOLD:
            consecutive += 1
            alerts.append({'frame': frame_num, 'prob': prob})

            if consecutive >= CONSECUTIVE_FRAMES and not confirmed:
                confirmed     = True
                confirm_frame = frame_num
                severity, _   = get_severity(prob)
                print(f"  🚨 ACCIDENT CONFIRMED at frame {frame_num} | "
                      f"Confidence: {prob:.2%} | Severity: {severity}")
        else:
            consecutive = 0   # reset streak on clean frame
            # Keep confirmed=True until video ends (don't reset mid-video)

        # ── Draw HUD ──────────────────────────────────────────────
        frame = draw_overlay(frame, pred, prob, consecutive,
                             confirmed, len(alerts), frame_num)

        cv2.imshow("Accident Detection", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("  Quit by user.")
            break

    video.release()
    cv2.destroyAllWindows()

    # ── Post-run summary ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("        ACCIDENT DETECTION SUMMARY")
    print("=" * 55)

    if not alerts:
        print("  ✅  No accident detected in video.")
    else:
        max_prob  = max(a['prob'] for a in alerts)
        severity, _ = get_severity(max_prob)
        print(f"  🚨  ACCIDENT DETECTED")
        print(f"  Severity      : {severity}")
        print(f"  Peak Confidence : {max_prob:.2%}")
        print(f"  Confirmed at  : Frame {confirm_frame}")
        print(f"  Alert frames  : {len(alerts)}")
        print(f"  Total frames  : {frame_num}")

    print("=" * 55)


if __name__ == '__main__':
    startapplication()