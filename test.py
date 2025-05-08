import cv2
import time
import numpy as np
import sys
import traceback

def try_camera_with_backend(index, backend=None):
    """Try to open a camera with specified backend"""
    try:
        if backend is None:
            print(f"\nAttempting to open camera {index} with default backend...")
            cap = cv2.VideoCapture(index)
        else:
            print(f"\nAttempting to open camera {index} with backend {backend}...")
            cap = cv2.VideoCapture(index, backend)
            
        if not cap.isOpened():
            print(f"  Failed to open camera {index}")
            return None
            
        print(f"  Successfully opened camera {index}!")
        
        # Try to read a frame
        print("  Attempting to read a frame...")
        ret, frame = cap.read()
        
        if not ret:
            print("  Failed to read frame!")
            cap.release()
            return None
            
        print(f"  Successfully read frame! Shape: {frame.shape}")
        
        # Try continuous reading for 2 seconds
        print("  Testing continuous reading for 2 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret:
                print("  Warning: Frame drop during continuous reading")
            else:
                frame_count += 1
                
        fps = frame_count / 2
        print(f"  Continuous reading test complete. Read {frame_count} frames in 2 seconds ({fps:.1f} FPS)")
        
        # Return the working VideoCapture object
        return cap
    except Exception as e:
        print(f"  Error: {e}")
        traceback.print_exc()
        return None

def test_all_backends():
    """Test all available camera backends in OpenCV"""
    # List of backends to try
    backends = [
        (cv2.CAP_ANY, "CAP_ANY (auto-detect)"),
        (cv2.CAP_DSHOW, "CAP_DSHOW (DirectShow)"),
        (cv2.CAP_MSMF, "CAP_MSMF (Microsoft Media Foundation)"),
        (cv2.CAP_V4L2, "CAP_V4L2 (Video4Linux2)"),
        (None, "Default backend")
    ]
    
    # List of camera indices to try
    camera_indices = [0, 1]
    
    best_cap = None
    best_backend_name = None
    best_camera_index = None
    
    print("===== CAMERA TROUBLESHOOTING TOOL =====")
    print("This tool will test various camera access methods to find what works on your system.")
    
    for index in camera_indices:
        for backend, backend_name in backends:
            try:
                if backend is None:
                    cap = try_camera_with_backend(index)
                else:
                    cap = try_camera_with_backend(index, backend)
                    
                if cap is not None:
                    # If we found a working camera, store it
                    if best_cap is None:
                        best_cap = cap
                        best_backend_name = backend_name
                        best_camera_index = index
                    else:
                        # We already have a working camera, release this one
                        cap.release()
                        
            except Exception as e:
                print(f"  Error testing {backend_name} with camera {index}: {e}")
    
    # If we found a working camera, display a test window
    if best_cap is not None:
        print("\n===== WORKING CAMERA CONFIGURATION FOUND =====")
        print(f"Camera index: {best_camera_index}")
        print(f"Backend: {best_backend_name}")
        print("\nDisplaying camera feed. Press 'q' to quit.")
        
        # Print usage advice for your main program
        print("\nTo fix your main program, use this configuration:")
        if best_backend_name == "Default backend":
            print(f"cap = cv2.VideoCapture({best_camera_index})")
        else:
            backend_var = [b for b, name in backends if name == best_backend_name][0]
            print(f"cap = cv2.VideoCapture({best_camera_index}, {backend_var.__str__()})")
        
        # Show the camera feed for 10 seconds
        try:
            start_time = time.time()
            while time.time() - start_time < 30:  # Run for 30 seconds
                ret, frame = best_cap.read()
                if ret:
                    # Add text showing the working configuration
                    text = f"Camera: {best_camera_index}, Backend: {best_backend_name}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to exit", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Working Camera Configuration", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Error reading frame during display")
                    break
        except Exception as e:
            print(f"Error during display: {e}")
        finally:
            best_cap.release()
            cv2.destroyAllWindows()
    else:
        print("\nNo working camera configuration found!")
        print("Possible solutions:")
        print("1. Check if your webcam is properly connected")
        print("2. Make sure no other application is using your camera")
        print("3. Try restarting your computer")
        print("4. Check your camera drivers and update if necessary")

if __name__ == "__main__":
    try:
        test_all_backends()
    except Exception as e:
        print(f"Error in main program: {e}")
        traceback.print_exc()