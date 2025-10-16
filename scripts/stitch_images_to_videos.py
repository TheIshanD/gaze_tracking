import os
import cv2
import argparse

def stitch_images_to_video(folder_path: str, frame_rate: float):
    # Resolve to absolute path
    folder_path = os.path.abspath(folder_path)
    print(f"\n📂 Using folder: {folder_path}")

    # Get all .jpg files in folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    if not image_files:
        print(f"❌ No .jpg images found in {folder_path}")
        return

    # Sort lexicographically
    image_files = sorted(image_files)
    print(f"🖼 Found {len(image_files)} image(s)")

    # Read first image to get frame size
    first_image_path = os.path.join(folder_path, image_files[0])
    first_frame = cv2.imread(first_image_path)

    if first_frame is None:
        print(f"❌ Error reading the first image: {first_image_path}")
        return

    height, width, _ = first_frame.shape
    size = (width, height)
    print(f"✅ Frame size detected: {width}x{height}")

    # Try multiple codecs for compatibility
    output_path_mp4 = os.path.join(folder_path, "prediction_validation_overlay.mp4")
    output_path_avi = os.path.join(folder_path, "prediction_validation_overlay.avi")

    # Try mp4v first, fallback to MJPG if unsupported
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better macOS compatibility than mp4v
    print(f"🎥 Using codec: avc1")

    out = cv2.VideoWriter(output_path_mp4, fourcc, frame_rate, size)
    if not out.isOpened():
        print("⚠️ Failed to open MP4 writer. Trying MJPG (.avi)...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path_avi, fourcc, frame_rate, size)
        output_path = output_path_avi
    else:
        output_path = output_path_mp4

    print(f"Creating video at {output_path} with frame rate {frame_rate} fps ...")

    written_frames = 0
    for i, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        # Check frame properties
        if frame.dtype != 'uint8':
            print(f"⚠️ Converting frame {img_name} to uint8 from {frame.dtype}")
            frame = cv2.convertScaleAbs(frame)

        if frame.shape[:2] != (height, width):
            print(f"⚠️ Resizing frame {img_name} from {frame.shape[1]}x{frame.shape[0]} to {width}x{height}")
            frame = cv2.resize(frame, size)
            
        out.write(frame)
        written_frames += 1

        # Print progress every 50 frames
        if i % 50 == 0 or i == len(image_files):
            print(f"  → Wrote frame {i}/{len(image_files)} ({img_name})")

    out.release()

    # Verify output file size
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ Done! Wrote {written_frames} frame(s)")
        print(f"📦 Output video saved to: {output_path}")
        print(f"📏 Video file size: {file_size_mb:.2f} MB")
    else:
        print("❌ No output video was created!")

    # Diagnostics summary
    print("\n--- Diagnostics Summary ---")
    print(f"  Total input images: {len(image_files)}")
    print(f"  Successfully written: {written_frames}")
    print(f"  Output resolution: {width}x{height}")
    print(f"  Frame rate: {frame_rate} fps")
    print(f"  Codec used: {'avc1' if 'mp4' in output_path else 'MJPG'}")
    print("-----------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Stitch .jpg images into a video.")
    parser.add_argument("folder", help="Relative path to the folder containing .jpg images.")
    parser.add_argument("frame_rate", type=float, help="Frame rate (can be a decimal).")
    args = parser.parse_args()

    stitch_images_to_video(args.folder, args.frame_rate)


if __name__ == "__main__":
    main()
