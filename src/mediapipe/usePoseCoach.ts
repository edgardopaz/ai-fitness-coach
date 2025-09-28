import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

export type SupportedMode = "squat" | "plank";

const SUPPORTED_MODES: readonly SupportedMode[] = ["squat", "plank"];

const SPEECH_COOLDOWN_MS = 3000;
const STANDING_KNEE_ANGLE = 165;
const BOTTOM_KNEE_ANGLE = 100;
const MIN_HIP_DELTA_FOR_REP = 0.07;
const MIN_KNEE_WIDTH_RATIO = 0.75;
const HIP_RETURN_TOLERANCE = 0.018;
const HIP_BELOW_KNEE_THRESHOLD = 0.02;
const FRONT_KNEE_ANGLE_MAX = 125;

const VISION_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

type SquatPhase = "start" | "up" | "down";
type LandmarkList = ReadonlyArray<NormalizedLandmark>;
type Point2D = readonly [number, number];
type SpeakOptions = { immediate?: boolean };
type FeedbackOptions = { immediate?: boolean; allowRepeat?: boolean; silent?: boolean };

export interface UsePoseCoachArgs {
  mode: SupportedMode | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
}

export interface UsePoseCoachResult {
  isModeSupported: boolean;
  isModelReady: boolean;
  isStarted: boolean;
  isVideoReady: boolean;
  repCount: number;
  feedback: string;
  stateLabel: string;
  appError: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

const stopMediaStream = (stream: MediaStream | null | undefined) => {
  if (!stream) {
    return;
  }
  stream.getTracks().forEach((track) => track.stop());
};

const getPoint = (landmarks: LandmarkList, index: number): Point2D | null => {
  const landmark = landmarks[index];
  if (!landmark) {
    return null;
  }
  return [landmark.x, landmark.y];
};

const angleBetween = (a: Point2D, b: Point2D, c: Point2D) => {
  const angle =
    Math.atan2(c[1] - b[1], c[0] - b[0]) -
    Math.atan2(a[1] - b[1], a[0] - b[0]);
  let degrees = Math.abs((angle * 180) / Math.PI);
  if (degrees > 180) {
    degrees = 360 - degrees;
  }
  return degrees;
};

const isSupportedMode = (mode: SupportedMode | null): mode is SupportedMode => {
  return mode != null && SUPPORTED_MODES.includes(mode);
};

export function usePoseCoach({ mode, videoRef, canvasRef }: UsePoseCoachArgs): UsePoseCoachResult {
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastSpeechTime = useRef(0);
  const standingHipRef = useRef<number | null>(null);
  const bottomHipRef = useRef<number | null>(null);
  const standingHipWidthRef = useRef<number | null>(null);

  const [isModelReady, setIsModelReady] = useState(false);
  const [isStarted, setIsStarted] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [squatState, setSquatState] = useState<SquatPhase>("start");
  const [feedback, setFeedback] = useState("Loading pose intelligence...");
  const [appError, setAppError] = useState<string | null>(null);

  const speak = useCallback(
    (text: string, options: SpeakOptions = {}) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        return;
      }

      const immediate = options.immediate ?? false;
      const now = Date.now();

      if (!immediate && now - lastSpeechTime.current < SPEECH_COOLDOWN_MS) {
        return;
      }

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
      lastSpeechTime.current = now;
    },
    []
  );

  const setFeedbackMessage = useCallback(
    (message: string, options: FeedbackOptions = {}) => {
      setFeedback((previous) => {
        const changed = previous !== message;
        if (changed) {
          if (!options.silent) {
            speak(message, { immediate: options.immediate });
          }
          return message;
        }

        if (!options.silent && options.allowRepeat) {
          speak(message, { immediate: options.immediate });
        }

        return previous;
      });
    },
    [speak]
  );

  const analyzeSquat = useCallback(
    (landmarks: LandmarkList) => {
      const rightHip = getPoint(landmarks, 24);
      const rightKnee = getPoint(landmarks, 26);
      const rightAnkle = getPoint(landmarks, 28);
      const leftHip = getPoint(landmarks, 23);
      const leftKnee = getPoint(landmarks, 25);
      const leftAnkle = getPoint(landmarks, 27);

      if (!rightHip && !leftHip) {
        return;
      }

      const computeKneeAngle = (hip: Point2D | null, knee: Point2D | null, ankle: Point2D | null) => {
        if (!hip || !knee || !ankle) {
          return null;
        }
        return angleBetween(hip, knee, ankle);
      };

      const rightAngle = computeKneeAngle(rightHip, rightKnee, rightAnkle);
      const leftAngle = computeKneeAngle(leftHip, leftKnee, leftAnkle);
      const kneeAngles = [rightAngle, leftAngle].filter((value): value is number => typeof value === "number");
      const kneeAngle =
        kneeAngles.length > 0 ? kneeAngles.reduce((sum, value) => sum + value, 0) / kneeAngles.length : null;

      const hipYValues = [rightHip, leftHip].filter(Boolean).map((point) => point![1]);
      const kneeYValues = [rightKnee, leftKnee].filter(Boolean).map((point) => point![1]);

      if (!hipYValues.length || !kneeYValues.length) {
        return;
      }

      const avgHipY = hipYValues.reduce((sum, value) => sum + value, 0) / hipYValues.length;
      const avgKneeY = kneeYValues.reduce((sum, value) => sum + value, 0) / kneeYValues.length;
      const hipVerticalDiff = avgHipY - avgKneeY;
      const hipWidth = rightHip && leftHip ? Math.abs(rightHip[0] - leftHip[0]) : null;
      const kneeWidth = rightKnee && leftKnee ? Math.abs(rightKnee[0] - leftKnee[0]) : null;
      const baseHipWidth = standingHipWidthRef.current ?? hipWidth;
      const kneeWidthRatio =
        baseHipWidth && baseHipWidth > 0 && kneeWidth !== null ? kneeWidth / baseHipWidth : null;
      const kneesWideEnough = kneeWidthRatio !== null && kneeWidthRatio >= MIN_KNEE_WIDTH_RATIO;
      const kneesWideBottom =
        kneesWideEnough && kneeAngle !== null && kneeAngle <= FRONT_KNEE_ANGLE_MAX;

      if (standingHipRef.current === null) {
        standingHipRef.current = avgHipY;
        if (hipWidth !== null && hipWidth > 0 && standingHipWidthRef.current === null) {
          standingHipWidthRef.current = hipWidth;
        }
      }

      const depthFromTop = standingHipRef.current !== null ? avgHipY - standingHipRef.current : 0;

      const kneeAngleIndicatesStanding = kneeAngle !== null && kneeAngle >= STANDING_KNEE_ANGLE;
      const hipReturn =
        Math.abs(depthFromTop) <= Math.max(HIP_RETURN_TOLERANCE, MIN_HIP_DELTA_FOR_REP * 0.35);
      const hipPositionIndicatesStanding = hipReturn || hipVerticalDiff <= 0;
      const isStanding = kneeAngleIndicatesStanding && hipPositionIndicatesStanding;

      const kneeAngleIndicatesBottom = kneeAngle !== null && kneeAngle <= BOTTOM_KNEE_ANGLE;
      const depthSufficient = depthFromTop >= MIN_HIP_DELTA_FOR_REP;
      const hipBelowKneeStrong = hipVerticalDiff >= HIP_BELOW_KNEE_THRESHOLD;
      const bottomDetected = depthSufficient && (kneeAngleIndicatesBottom || hipBelowKneeStrong || kneesWideBottom);

      if (isStanding) {
        if (
          squatState === "down" &&
          bottomHipRef.current !== null &&
          standingHipRef.current !== null &&
          bottomHipRef.current - standingHipRef.current >= MIN_HIP_DELTA_FOR_REP
        ) {
          setRepCount((previous) => {
            const next = previous + 1;
            speak(`Rep ${next}`, { immediate: true });
            return next;
          });
          setFeedbackMessage("Stand tall and lock the rep before the next drive.");
        } else {
          setFeedbackMessage("Brace, set your stance, and control the next descent.");
        }
        setSquatState("up");
        bottomHipRef.current = null;
        return;
      }

      if (bottomDetected) {
        if (squatState !== "down") {
          setSquatState("down");
          bottomHipRef.current = avgHipY;
          if (hipVerticalDiff < HIP_BELOW_KNEE_THRESHOLD) {
            setFeedbackMessage("Drop another inch so hips finish just below the knees.", {
              immediate: true,
            });
          } else {
            setFeedbackMessage("Great depth. Stay tight and drive up through your heels.", {
              immediate: true,
            });
          }
        }
        return;
      }

      if (squatState === "down") {
        setFeedbackMessage("Drive through your heels and finish tall at the top.");
      } else {
        setFeedbackMessage("Control the descent - hips back, knees tracking over toes.");
      }
    },
    [squatState, speak, setFeedbackMessage]
  );

  const analyzePlank = useCallback(
    (landmarks: LandmarkList) => {
      const shoulder = getPoint(landmarks, 12);
      const hip = getPoint(landmarks, 24);
      const ankle = getPoint(landmarks, 28);
      if (!shoulder || !hip || !ankle) {
        return;
      }

      const bodyAngle = angleBetween(shoulder, hip, ankle);
      if (bodyAngle < 165) {
        setFeedbackMessage("Lock in a straight line from ears to heels.", { immediate: true });
      } else {
        setFeedbackMessage("Strong plank - stay long through your spine and keep breathing.");
      }
    },
    [setFeedbackMessage]
  );

  useEffect(() => {
    let isCancelled = false;

    const initializeModel = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);
        if (isCancelled) {
          return;
        }
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: POSE_MODEL_URL,
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (isCancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setIsModelReady(true);
        setFeedback("Coach ready. Press start to begin.");
      } catch (error) {
        if (!isCancelled) {
          const message = error instanceof Error ? error.message : "Failed to initialize pose detection.";
          setAppError(message);
        }
      }
    };

    initializeModel();

    return () => {
      isCancelled = true;
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
    };
  }, []);

  const stop = useCallback(() => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;

    if (videoElement) {
      stopMediaStream(videoElement.srcObject as MediaStream | null);
      videoElement.srcObject = null;
      videoElement.pause();
    }

    lastVideoTimeRef.current = -1;
    standingHipRef.current = null;
    bottomHipRef.current = null;
    standingHipWidthRef.current = null;

    if (canvasElement) {
      const context = canvasElement.getContext("2d");
      if (context) {
        context.clearRect(0, 0, canvasElement.width, canvasElement.height);
      }
    }

    setIsStarted(false);
    setIsVideoReady(false);
    setFeedbackMessage("Session paused. Hit start when you want the coach back in.");
  }, [canvasRef, videoRef, setFeedbackMessage]);

  const start = useCallback(async () => {
    if (!isSupportedMode(mode)) {
      stop();
      setAppError(null);
      setFeedbackMessage("Select squat or plank to run live MediaPipe tracking.", { immediate: true });
      return;
    }

    const videoElement = videoRef.current;
    if (!videoElement) {
      setAppError("Video element is not available.");
      return;
    }

    if (!landmarkerRef.current) {
      setAppError("Pose model is still loading. Please try again in a moment.");
      return;
    }

    try {
      setAppError(null);
      setFeedbackMessage("Setting up your camera...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });

      stopMediaStream(videoElement.srcObject as MediaStream | null);
      videoElement.srcObject = stream;

      await new Promise<void>((resolve, reject) => {
        videoElement.onloadedmetadata = () => {
          videoElement.onloadedmetadata = null;
          videoElement.onerror = null;
          resolve();
        };
        videoElement.onerror = () => {
          videoElement.onloadedmetadata = null;
          videoElement.onerror = null;
          reject(new Error("Unable to load video stream."));
        };
      });

      await videoElement.play();
      lastVideoTimeRef.current = -1;
      setIsVideoReady(true);
      setIsStarted(true);
      setFeedbackMessage(
        mode === "squat"
          ? "Tracking squat mechanics - sit back, stay tall, and drive the floor away."
          : "Tracking plank alignment - reach long, pack shoulders, and breathe."
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to access the camera.";
      setAppError(message);
      setIsVideoReady(false);
      setIsStarted(false);
      if (videoElement) {
        stopMediaStream(videoElement.srcObject as MediaStream | null);
        videoElement.srcObject = null;
      }
      setFeedbackMessage("Camera unavailable. Check permissions and try again.");
    }
  }, [mode, setFeedbackMessage, videoRef, stop]);

  useEffect(() => {
    if (!isSupportedMode(mode)) {
      if (isStarted) {
        stop();
      }
      setRepCount(0);
      setSquatState("start");
      standingHipRef.current = null;
      bottomHipRef.current = null;
      standingHipWidthRef.current = null;
      setFeedback("MediaPipe tracking is available for squat and plank labs.");
      setAppError(null);
      return;
    }

    setRepCount(0);
    setSquatState("start");
    standingHipRef.current = null;
    bottomHipRef.current = null;
    standingHipWidthRef.current = null;
    setAppError(null);

    if (!isStarted) {
      setFeedback(
        mode === "squat"
          ? "Coach ready: brace tall, knees track over toes, and sit back confidently."
          : "Coach ready: press the floor away, lengthen through the crown, and lock the core."
      );
    }
  }, [isStarted, mode, stop]);

  useEffect(() => {
    if (!isStarted || !isVideoReady || !isSupportedMode(mode)) {
      return;
    }

    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!videoElement || !canvasElement || !landmarker) {
      return;
    }

    const context = canvasElement.getContext("2d");
    if (!context) {
      return;
    }

    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;

    const drawingUtils = new DrawingUtils(context);
    let animationFrame = 0;

    const renderLoop = () => {
      if (!landmarkerRef.current) {
        animationFrame = requestAnimationFrame(renderLoop);
        return;
      }

      const currentTime = videoElement.currentTime;
      if (currentTime === lastVideoTimeRef.current) {
        animationFrame = requestAnimationFrame(renderLoop);
        return;
      }
      lastVideoTimeRef.current = currentTime;

      try {
        context.save();
        context.clearRect(0, 0, canvasElement.width, canvasElement.height);

        const results: PoseLandmarkerResult = landmarkerRef.current.detectForVideo(videoElement, performance.now());
        const poseLandmarks = results.landmarks?.[0];

        if (poseLandmarks && poseLandmarks.length) {
          drawingUtils.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
            lineWidth: 3,
            color: "rgba(147, 51, 234, 0.7)",
          });
          drawingUtils.drawLandmarks(poseLandmarks, { radius: 3, color: "#38bdf8" });

          if (mode === "squat") {
            analyzeSquat(poseLandmarks);
          } else {
            analyzePlank(poseLandmarks);
          }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Pose detection failed.";
        setAppError(message);
      } finally {
        context.restore();
      }

      animationFrame = requestAnimationFrame(renderLoop);
    };

    animationFrame = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animationFrame);
      context.clearRect(0, 0, canvasElement.width, canvasElement.height);
      lastVideoTimeRef.current = -1;
    };
  }, [isStarted, isVideoReady, mode, analyzePlank, analyzeSquat, videoRef, canvasRef]);

  useEffect(() => {
    const videoElement = videoRef.current;
    return () => {
      if (videoElement) {
        stopMediaStream(videoElement.srcObject as MediaStream | null);
        videoElement.srcObject = null;
      }
    };
  }, [videoRef]);

  const stateLabel = (() => {
    if (!isSupportedMode(mode)) {
      return "N/A";
    }

    if (!isStarted) {
      return "SET";
    }

    if (mode === "plank") {
      return "HOLD";
    }

    switch (squatState) {
      case "down":
        return "LOW";
      case "up":
        return "UP";
      default:
        return "SET";
    }
  })();

  return {
    isModeSupported: isSupportedMode(mode),
    isModelReady,
    isStarted,
    isVideoReady,
    repCount,
    feedback,
    stateLabel,
    appError,
    start,
    stop,
  };
}
