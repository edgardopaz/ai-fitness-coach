import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

export type SupportedMode = "squat" | "plank" | "pushup" | "pullup" | "jumpingJack";

const SUPPORTED_MODES: readonly SupportedMode[] = [
  "squat",
  "plank",
  "pushup",
  "pullup",
  "jumpingJack",
];

const SPEECH_COOLDOWN_MS = 3000;
const STANDING_KNEE_ANGLE = 165;
const BOTTOM_KNEE_ANGLE = 100;
const MIN_HIP_DELTA_FOR_REP = 0.07;
const MIN_KNEE_WIDTH_RATIO = 0.75;
const HIP_RETURN_TOLERANCE = 0.018;
const HIP_BELOW_KNEE_THRESHOLD = 0.02;
const FRONT_KNEE_ANGLE_MAX = 125;

const STANDING_ANGLE_RELAXATION = 3;
const BOTTOM_ANGLE_RELAXATION = 5;
const REP_DEPTH_MARGIN = 0.9;
const HIP_BELOW_THRESHOLD_MARGIN = 0.8;
const MIN_BOTTOM_HOLD_FRAMES = 2;
const MIN_STANDING_HOLD_FRAMES = 2;
const SMOOTHING_ALPHA_HIPS = 0.25;
const SMOOTHING_ALPHA_KNEE = 0.35;
const SMOOTHING_ALPHA_PLANK = 0.25;
const PLANK_WARN_FRAMES = 6;
const PLANK_STRICT_FRAMES = 12;
const PLANK_MAX_SHOULDER_HIP_DELTA = 0.16;
const PLANK_MAX_HIP_ANKLE_DELTA = 0.22;
const PLANK_STANDING_GRACE_FRAMES = 4;
const PLANK_STRONG_ANGLE = 165;
const PLANK_MIN_ANGLE = 158;
const PLANK_TIMER_UPDATE_INTERVAL_MS = 100;

const PUSHUP_TOP_ANGLE = 148;
const PUSHUP_BOTTOM_ANGLE = 110;

const PULLUP_TOP_NOSE_OFFSET = 0.04;
const PULLUP_BOTTOM_WRIST_OFFSET = 0.14;

const JACK_DEFAULT_NEUTRAL_LEG_RATIO = 0.68;
const JACK_DEFAULT_NEUTRAL_WRIST_RATIO = 0.82;
const JACK_MIN_NEUTRAL_ANKLE_GAP = 0.08;
const JACK_MIN_NEUTRAL_WRIST_GAP = 0.18;
const JACK_WIDE_LEG_EXTRA = 0.09;
const JACK_WIDE_LEG_RATIO = 0.45;
const JACK_CENTER_LEG_EXTRA = 0.05;
const JACK_CENTER_LEG_RATIO = 0.25;
const JACK_ARM_OVERHEAD_DELTA = 0.04;
const JACK_ARM_DOWN_DELTA = -0.035;
const JACK_ALMOST_ARM_RATIO = 0.8;
const JACK_ALMOST_LEG_RATIO = 0.85;
const JACK_CENTER_RETURN_RATIO = 1.25;
const JACK_FEEDBACK_COOLDOWN_MS = 1200;
const VISION_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

type SquatPhase = "start" | "up" | "down";
type PushupPhase = "setup" | "lowering" | "press";
type PullupPhase = "hang" | "pull" | "top";
type JackPhase = "center" | "wide";
type PlankPhase = "setup" | "hold" | "adjust";
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
  plankHoldMs: number;
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

const distance = (a: Point2D, b: Point2D) => {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
};

const smoothValue = (ref: { current: number | null }, value: number | null, alpha: number) => {
  if (value === null || Number.isNaN(value)) {
    return null;
  }
  if (ref.current === null || Number.isNaN(ref.current)) {
    ref.current = value;
  } else {
    ref.current = ref.current + alpha * (value - ref.current);
  }
  return ref.current;
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
  const smoothedHipYRef = useRef<number | null>(null);
  const smoothedKneeYRef = useRef<number | null>(null);
  const smoothedKneeAngleRef = useRef<number | null>(null);
  const standingHoldFramesRef = useRef(0);
  const bottomHoldFramesRef = useRef(0);
  const plankAngleRef = useRef<number | null>(null);
  const plankLowFramesRef = useRef(0);
  const pushupBottomSeenRef = useRef(false);
  const pullupHangSeenRef = useRef(false);
  const jackCenterReadyRef = useRef(true);
  const lastJackFeedbackRef = useRef(0);
  const jackNeutralAnkleGapRef = useRef<number | null>(null);
  const jackNeutralWristGapRef = useRef<number | null>(null);
  const jackDebugCounterRef = useRef(0);
  const plankStandingFramesRef = useRef(0);
  const plankHoldMsRef = useRef(0);
  const plankLastTimestampRef = useRef<number | null>(null);
  const plankHoldActiveRef = useRef(false);
  const lastPlankBroadcastRef = useRef(0);

  const [isModelReady, setIsModelReady] = useState(false);
  const [isStarted, setIsStarted] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [feedback, setFeedback] = useState("Loading pose intelligence...");
  const [appError, setAppError] = useState<string | null>(null);
  const [squatState, setSquatState] = useState<SquatPhase>("start");
  const [pushupState, setPushupState] = useState<PushupPhase>("setup");
  const [pullupState, setPullupState] = useState<PullupPhase>("hang");
  const [jackState, setJackState] = useState<JackPhase>("center");
  const [plankState, setPlankState] = useState<PlankPhase>("setup");
  const [plankHoldMs, setPlankHoldMs] = useState(0);

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

  const clearPoseRefs = useCallback(() => {
    standingHipRef.current = null;
    bottomHipRef.current = null;
    standingHipWidthRef.current = null;
    smoothedHipYRef.current = null;
    smoothedKneeYRef.current = null;
    smoothedKneeAngleRef.current = null;
    standingHoldFramesRef.current = 0;
    bottomHoldFramesRef.current = 0;
    plankAngleRef.current = null;
    plankLowFramesRef.current = 0;
    plankHoldMsRef.current = 0;
    plankLastTimestampRef.current = null;
    plankHoldActiveRef.current = false;
    lastPlankBroadcastRef.current = 0;
    pushupBottomSeenRef.current = false;
    pullupHangSeenRef.current = false;
    jackCenterReadyRef.current = true;
    jackNeutralAnkleGapRef.current = null;
    jackNeutralWristGapRef.current = null;
    lastJackFeedbackRef.current = 0;
    jackDebugCounterRef.current = 0;
    plankStandingFramesRef.current = 0;

    setRepCount(0);
    setSquatState("start");
    setPushupState("setup");
    setPullupState("hang");
    setJackState("center");
    setPlankState("setup");
    setPlankHoldMs(0);
  }, []);

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
      const rawKneeAngle =
        kneeAngles.length > 0 ? kneeAngles.reduce((sum, value) => sum + value, 0) / kneeAngles.length : null;

      const hipYValues = [rightHip, leftHip].filter(Boolean).map((point) => point![1]);
      const kneeYValues = [rightKnee, leftKnee].filter(Boolean).map((point) => point![1]);

      if (!hipYValues.length || !kneeYValues.length) {
        return;
      }

      const avgHipY = hipYValues.reduce((sum, value) => sum + value, 0) / hipYValues.length;
      const avgKneeY = kneeYValues.reduce((sum, value) => sum + value, 0) / kneeYValues.length;

      const hipY = smoothValue(smoothedHipYRef, avgHipY, SMOOTHING_ALPHA_HIPS) ?? avgHipY;
      const kneeY = smoothValue(smoothedKneeYRef, avgKneeY, SMOOTHING_ALPHA_HIPS) ?? avgKneeY;
      const kneeAngle =
        rawKneeAngle !== null
          ? smoothValue(smoothedKneeAngleRef, rawKneeAngle, SMOOTHING_ALPHA_KNEE) ?? rawKneeAngle
          : null;
      const effectiveKneeAngle = kneeAngle ?? rawKneeAngle;

      const hipVerticalDiff = hipY - kneeY;
      const hipWidth = rightHip && leftHip ? Math.abs(rightHip[0] - leftHip[0]) : null;
      const kneeWidth = rightKnee && leftKnee ? Math.abs(rightKnee[0] - leftKnee[0]) : null;
      const baseHipWidth = standingHipWidthRef.current ?? hipWidth;
      const kneeWidthRatio =
        baseHipWidth && baseHipWidth > 0 && kneeWidth !== null ? kneeWidth / baseHipWidth : null;
      const kneesWideEnough = kneeWidthRatio !== null && kneeWidthRatio >= MIN_KNEE_WIDTH_RATIO * 0.9;
      const kneesWideBottom =
        kneesWideEnough && effectiveKneeAngle !== null && effectiveKneeAngle <= FRONT_KNEE_ANGLE_MAX + 4;

      if (standingHipRef.current === null) {
        standingHipRef.current = hipY;
      }
      if (standingHipWidthRef.current === null && hipWidth !== null && hipWidth > 0) {
        standingHipWidthRef.current = hipWidth;
      }

      const depthFromTop = standingHipRef.current !== null ? hipY - standingHipRef.current : 0;
      const kneeAngleIndicatesStanding =
        effectiveKneeAngle !== null && effectiveKneeAngle >= STANDING_KNEE_ANGLE - STANDING_ANGLE_RELAXATION;
      const hipReturn =
        Math.abs(depthFromTop) <= Math.max(HIP_RETURN_TOLERANCE * 1.5, MIN_HIP_DELTA_FOR_REP * 0.35);
      const hipPositionIndicatesStanding = hipReturn || hipVerticalDiff <= 0;
      const isStanding = kneeAngleIndicatesStanding && hipPositionIndicatesStanding;

      const kneeAngleIndicatesBottom =
        effectiveKneeAngle !== null && effectiveKneeAngle <= BOTTOM_KNEE_ANGLE + BOTTOM_ANGLE_RELAXATION;
      const repDepthThreshold = MIN_HIP_DELTA_FOR_REP * REP_DEPTH_MARGIN;
      const depthSufficient = depthFromTop >= repDepthThreshold;
      const hipBelowKneeStrong = hipVerticalDiff >= HIP_BELOW_KNEE_THRESHOLD * HIP_BELOW_THRESHOLD_MARGIN;
      const bottomDetected = depthSufficient && (kneeAngleIndicatesBottom || hipBelowKneeStrong || kneesWideBottom);

      if (isStanding) {
        standingHoldFramesRef.current = Math.min(standingHoldFramesRef.current + 1, 60);
        bottomHoldFramesRef.current = 0;

        if (standingHoldFramesRef.current >= MIN_STANDING_HOLD_FRAMES) {
          if (
            squatState === "down" &&
            bottomHipRef.current !== null &&
            standingHipRef.current !== null &&
            bottomHipRef.current - standingHipRef.current >= repDepthThreshold
          ) {
            setRepCount((previous) => {
              const next = previous + 1;
              speak(`Rep ${next}`, { immediate: true });
              return next;
            });
            setFeedbackMessage("Stand tall and lock the rep before the next drive.", { silent: true });
          } else if (squatState !== "up") {
            setFeedbackMessage("Brace, set your stance, and control the next descent.", { silent: true });
          }

          if (hipWidth !== null && hipWidth > 0) {
            standingHipWidthRef.current =
              standingHipWidthRef.current === null
                ? hipWidth
                : Math.min(standingHipWidthRef.current, hipWidth);
          }

          standingHipRef.current = standingHipRef.current === null ? hipY : Math.min(standingHipRef.current, hipY);
          bottomHipRef.current = null;
          if (squatState !== "up") {
            setSquatState("up");
          }
        }

        return;
      }

      standingHoldFramesRef.current = 0;

      if (bottomDetected) {
        bottomHoldFramesRef.current = Math.min(bottomHoldFramesRef.current + 1, 60);
      } else if (bottomHoldFramesRef.current > 0) {
        bottomHoldFramesRef.current -= 1;
      }

      if (bottomDetected && bottomHoldFramesRef.current >= MIN_BOTTOM_HOLD_FRAMES && squatState === "up") {
        bottomHipRef.current = hipY;
        setSquatState("down");
        if (!hipBelowKneeStrong) {
          setFeedbackMessage("Drop another inch so hips finish just below the knees.", {
            immediate: true,
          });
        } else {
          setFeedbackMessage("Great depth. Stay tight and drive up through your heels.", {
            immediate: true,
          });
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

  const analyzePushup = useCallback(
    (landmarks: LandmarkList) => {
      const shoulder = getPoint(landmarks, 12);
      const elbow = getPoint(landmarks, 14);
      const wrist = getPoint(landmarks, 16);

      if (!shoulder || !elbow || !wrist) {
        return;
      }

      const elbowAngle = angleBetween(shoulder, elbow, wrist);

      if (elbowAngle <= PUSHUP_BOTTOM_ANGLE) {
        pushupBottomSeenRef.current = true;
        if (pushupState !== "lowering") {
          setPushupState("lowering");
          setFeedbackMessage("Lower until chest hovers above the floor.", { immediate: true });
        }
        return;
      }

      if (elbowAngle >= PUSHUP_TOP_ANGLE) {
        if (pushupState === "lowering") {
          if (pushupBottomSeenRef.current) {
            setRepCount((previous) => {
              const next = previous + 1;
              speak(`Rep ${next}`, { immediate: true });
              return next;
            });
            setFeedbackMessage("Strong lockout - reset your brace for the next rep.", { silent: true });
          } else {
            setFeedbackMessage("Rep didn't count - sink a bit deeper before pressing up.", {
              immediate: true,
            });
          }
        }
        pushupBottomSeenRef.current = false;
        if (pushupState !== "press") {
          setPushupState("press");
        }
        return;
      }

      if (pushupState === "lowering" && !pushupBottomSeenRef.current && elbowAngle > PUSHUP_BOTTOM_ANGLE + 5) {
        setFeedbackMessage("Bend elbows more to reach full depth before pressing.", { allowRepeat: false });
      }

      if (pushupState !== "setup") {
        setPushupState("setup");
      }
    },
    [pushupState, setFeedbackMessage, speak]
  );

  const analyzePullup = useCallback(
    (landmarks: LandmarkList) => {
      const leftWrist = getPoint(landmarks, 15);
      const rightWrist = getPoint(landmarks, 16);
      const leftShoulder = getPoint(landmarks, 11);
      const rightShoulder = getPoint(landmarks, 12);
      const nose = getPoint(landmarks, 0);

      if (!leftWrist || !rightWrist || !leftShoulder || !rightShoulder || !nose) {
        return;
      }

      const avgShoulderY = (leftShoulder[1] + rightShoulder[1]) / 2;
      const avgWristY = (leftWrist[1] + rightWrist[1]) / 2;
      const hangDepth = avgWristY - avgShoulderY;

      if (hangDepth >= PULLUP_BOTTOM_WRIST_OFFSET) {
        pullupHangSeenRef.current = true;
        if (pullupState !== 'hang') {
          setPullupState('hang');
          setFeedbackMessage('Full hang - engage lats before the next pull.', { immediate: true });
        }
        return;
      }

      if (nose[1] <= avgShoulderY + PULLUP_TOP_NOSE_OFFSET) {
        if (pullupState === 'pull') {
          if (pullupHangSeenRef.current) {
            setRepCount((previous) => {
              const next = previous + 1;
              speak(`Rep ${next}`, { immediate: true });
              return next;
            });
            setFeedbackMessage("Nice pull - own the top before lowering.", { silent: true });
          } else {
            setFeedbackMessage("Rep didn't count - hit a full hang before pulling up.", { immediate: true });
          }
        }
        pullupHangSeenRef.current = false;
        if (pullupState !== 'top') {
          setPullupState('top');
        }
        return;
      }

      if (pullupState !== 'pull') {
        setPullupState('pull');
      } else if (!pullupHangSeenRef.current && hangDepth < PULLUP_BOTTOM_WRIST_OFFSET * 0.6) {
        setFeedbackMessage("Reset to a full hang so the rep will count.", { allowRepeat: false });
      }
    },
    [pullupState, setFeedbackMessage, speak]
  );

  const analyzeJumpingJack = useCallback(
    (landmarks: LandmarkList) => {
      const leftAnkle = getPoint(landmarks, 27);
      const rightAnkle = getPoint(landmarks, 28);
      const leftWrist = getPoint(landmarks, 15);
      const rightWrist = getPoint(landmarks, 16);
      const leftShoulder = getPoint(landmarks, 11);
      const rightShoulder = getPoint(landmarks, 12);
      const leftHip = getPoint(landmarks, 23);
      const rightHip = getPoint(landmarks, 24);

      if (!leftAnkle || !rightAnkle || !leftWrist || !rightWrist || !leftShoulder || !rightShoulder) {
        return;
      }

      const ankleGap = distance(leftAnkle, rightAnkle);
      const wristGap = distance(leftWrist, rightWrist);
      const avgShoulderY = (leftShoulder[1] + rightShoulder[1]) / 2;
      const avgWristY = (leftWrist[1] + rightWrist[1]) / 2;

      const shoulderWidthX = Math.max(Math.abs(leftShoulder[0] - rightShoulder[0]), 0.001);
      const hipWidthX =
        leftHip && rightHip ? Math.max(Math.abs(leftHip[0] - rightHip[0]), 0.001) : shoulderWidthX;

      const neutralLegBaseline =
        jackNeutralAnkleGapRef.current ??
        Math.max(JACK_MIN_NEUTRAL_ANKLE_GAP, hipWidthX * JACK_DEFAULT_NEUTRAL_LEG_RATIO);
      const neutralWristBaseline =
        jackNeutralWristGapRef.current ??
        Math.max(JACK_MIN_NEUTRAL_WRIST_GAP, shoulderWidthX * JACK_DEFAULT_NEUTRAL_WRIST_RATIO);

      const legWideThreshold = Math.max(
        neutralLegBaseline + JACK_WIDE_LEG_EXTRA,
        neutralLegBaseline + hipWidthX * JACK_WIDE_LEG_RATIO
      );
      const legCenterThreshold = Math.min(
        neutralLegBaseline + JACK_CENTER_LEG_EXTRA,
        neutralLegBaseline + hipWidthX * JACK_CENTER_LEG_RATIO
      );

      const leftArmDelta = leftShoulder[1] - leftWrist[1];
      const rightArmDelta = rightShoulder[1] - rightWrist[1];

      const armsOverhead =
        leftArmDelta >= JACK_ARM_OVERHEAD_DELTA && rightArmDelta >= JACK_ARM_OVERHEAD_DELTA;
      const armsDown =
        leftArmDelta <= JACK_ARM_DOWN_DELTA && rightArmDelta <= JACK_ARM_DOWN_DELTA;
      const armsAlmostOverhead =
        leftArmDelta >= JACK_ARM_OVERHEAD_DELTA * JACK_ALMOST_ARM_RATIO &&
        rightArmDelta >= JACK_ARM_OVERHEAD_DELTA * JACK_ALMOST_ARM_RATIO;
      const armsAlmostDown =
        leftArmDelta <= JACK_ARM_DOWN_DELTA * JACK_ALMOST_ARM_RATIO &&
        rightArmDelta <= JACK_ARM_DOWN_DELTA * JACK_ALMOST_ARM_RATIO;

      const legsWideEnough = ankleGap >= legWideThreshold;
      const legsAlmostWide = ankleGap >= legWideThreshold * JACK_ALMOST_LEG_RATIO;
      const legsTogetherEnough = ankleGap <= legCenterThreshold;
      const legsAlmostTogether = ankleGap <= legCenterThreshold * JACK_CENTER_RETURN_RATIO;
      const wristsAtSides = wristGap <= neutralWristBaseline * 1.2;

      const now = Date.now();
      const timeSinceFeedback = now - lastJackFeedbackRef.current;
      const canGiveFeedback = timeSinceFeedback > JACK_FEEDBACK_COOLDOWN_MS;

      const widePoseStrong = armsOverhead && legsWideEnough;
      const widePoseBorderline =
        !widePoseStrong &&
        ((armsOverhead && legsAlmostWide) ||
          (armsAlmostOverhead && legsWideEnough) ||
          (armsAlmostOverhead && legsAlmostWide));

      if (widePoseStrong || widePoseBorderline) {
        const wasReady = jackCenterReadyRef.current;
        const hasNeutralBaseline = jackNeutralAnkleGapRef.current !== null;

        if (wasReady && hasNeutralBaseline) {
          setRepCount((previous) => {
            const next = previous + 1;
            speak(`Rep ${next}`, { immediate: true });
            return next;
          });

          if (widePoseBorderline && canGiveFeedback) {
            const borderlineMessage =
              !armsOverhead && !legsWideEnough
                ? "Rep counted - reach higher and step wider for cleaner reps."
                : !armsOverhead
                ? "Rep counted - try raising arms a little higher."
                : !legsWideEnough
                ? "Rep counted - jump feet a little wider."
                : "Rep counted - keep the motion smooth.";
            setFeedbackMessage(borderlineMessage, { allowRepeat: false });
          } else {
            setFeedbackMessage("Rep counted - stay tall and control the landing.", { silent: true });
          }
        } else if (wasReady && !hasNeutralBaseline && canGiveFeedback) {
          setFeedbackMessage("Start neutral so reps count - feet together, arms down.", { immediate: true });
        } else if (!wasReady && canGiveFeedback) {
          setFeedbackMessage("Finish the reset before starting the next rep.", { immediate: true });
        }

        jackCenterReadyRef.current = false;

        if (jackState !== "wide") {
          setJackState("wide");
        }

        lastJackFeedbackRef.current = now;
        return;
      }

      if (armsDown && legsTogetherEnough && wristsAtSides) {
        jackCenterReadyRef.current = true;
        jackNeutralAnkleGapRef.current =
          jackNeutralAnkleGapRef.current === null
            ? ankleGap
            : jackNeutralAnkleGapRef.current * 0.65 + ankleGap * 0.35;
        jackNeutralWristGapRef.current =
          jackNeutralWristGapRef.current === null
            ? wristGap
            : jackNeutralWristGapRef.current * 0.65 + wristGap * 0.35;

        if (jackState !== "center") {
          setJackState("center");
        }

        lastJackFeedbackRef.current = now;
        return;
      }

      if (canGiveFeedback) {
        if (jackCenterReadyRef.current && (legsAlmostWide || armsAlmostOverhead)) {
          if (!armsOverhead && legsWideEnough) {
            setFeedbackMessage("Arms not overhead - reach all the way up.", { immediate: true });
            lastJackFeedbackRef.current = now;
          } else if (armsOverhead && !legsWideEnough) {
            setFeedbackMessage("Feet not wide enough - jump out farther.", { immediate: true });
            lastJackFeedbackRef.current = now;
          } else if (!armsOverhead && !legsWideEnough) {
            setFeedbackMessage("Go bigger - reach overhead and jump feet wider.", { immediate: true });
            lastJackFeedbackRef.current = now;
          }
        } else if (!jackCenterReadyRef.current && jackState === "wide" && (legsAlmostTogether || armsAlmostDown)) {
          if (!legsTogetherEnough && legsAlmostTogether) {
            setFeedbackMessage("Bring your feet together to finish the rep.", { immediate: true });
            lastJackFeedbackRef.current = now;
          } else if (!armsDown && armsAlmostDown) {
            setFeedbackMessage("Lower your arms to your sides before the next rep.", { immediate: true });
            lastJackFeedbackRef.current = now;
          } else if (!wristsAtSides && armsAlmostDown) {
            setFeedbackMessage("Keep arms close to your sides at the bottom.", { immediate: true });
            lastJackFeedbackRef.current = now;
          }
        }
      }

      if (import.meta.env.DEV) {
        jackDebugCounterRef.current = (jackDebugCounterRef.current + 1) % 10;
        if (jackDebugCounterRef.current === 0) {
          console.debug("[JumpingJack]", {
            leftAnkle,
            rightAnkle,
            leftWrist,
            rightWrist,
            leftShoulder,
            rightShoulder,
            leftHip,
            rightHip,
            ankleGap,
            wristGap,
            avgShoulderY,
            avgWristY,
            shoulderWidthX,
            hipWidthX,
            neutralLegBaseline,
            neutralWristBaseline,
            legWideThreshold,
            legCenterThreshold,
            leftArmDelta,
            rightArmDelta,
            armsOverhead,
            armsDown,
            armsAlmostOverhead,
            armsAlmostDown,
            legsWideEnough,
            legsAlmostWide,
            legsTogetherEnough,
            legsAlmostTogether,
            wristsAtSides,
            jackCenterReady: jackCenterReadyRef.current,
            jackState,
            timeSinceFeedback,
          });
        }
      }
    },
    [jackState, setFeedbackMessage, speak]
  );
  const analyzePlank = useCallback(
    (landmarks: LandmarkList, timestamp: number) => {
      const shoulder = getPoint(landmarks, 12);
      const hip = getPoint(landmarks, 24);
      const ankle = getPoint(landmarks, 28);
      if (!shoulder || !hip || !ankle) {
        return;
      }

      const shoulderHipDeltaY = Math.abs(shoulder[1] - hip[1]);
      const hipAnkleDeltaY = Math.abs(hip[1] - ankle[1]);

      if (shoulderHipDeltaY > PLANK_MAX_SHOULDER_HIP_DELTA || hipAnkleDeltaY > PLANK_MAX_HIP_ANKLE_DELTA) {
        plankStandingFramesRef.current = Math.min(plankStandingFramesRef.current + 1, 120);
        if (plankStandingFramesRef.current >= PLANK_STANDING_GRACE_FRAMES) {
          if (plankHoldActiveRef.current) {
            plankHoldActiveRef.current = false;
            plankLastTimestampRef.current = null;
            plankHoldMsRef.current = 0;
            lastPlankBroadcastRef.current = 0;
            setPlankHoldMs(0);
            setPlankState("setup");
            setFeedbackMessage("Timer paused - drop hips to shoulder height before holding.", { immediate: true });
          } else if (plankState !== "setup") {
            setPlankState("setup");
            setFeedbackMessage("Get into a straight plank before the timer starts.", { allowRepeat: false });
          }
        }
        return;
      }

      if (plankStandingFramesRef.current !== 0) {
        plankStandingFramesRef.current = 0;
      }

      const bodyAngleRaw = angleBetween(shoulder, hip, ankle);
      const bodyAngle = smoothValue(plankAngleRef, bodyAngleRaw, SMOOTHING_ALPHA_PLANK) ?? bodyAngleRaw;

      if (bodyAngle >= PLANK_STRONG_ANGLE) {
        plankLowFramesRef.current = Math.max(plankLowFramesRef.current - 1, 0);

        if (!plankHoldActiveRef.current) {
          plankHoldActiveRef.current = true;
          plankLastTimestampRef.current = timestamp;
          plankHoldMsRef.current = 0;
          lastPlankBroadcastRef.current = 0;
          if (plankState !== "hold") {
            setPlankState("hold");
          }
          setFeedbackMessage("Strong plank - stay long through your spine and keep breathing.", {
            silent: true,
          });
          setPlankHoldMs(0);
        } else if (plankLastTimestampRef.current !== null) {
          const delta = timestamp - plankLastTimestampRef.current;
          if (delta > 0) {
            plankHoldMsRef.current += delta;
            if (
              plankHoldMsRef.current - lastPlankBroadcastRef.current >= PLANK_TIMER_UPDATE_INTERVAL_MS ||
              plankHoldMsRef.current < lastPlankBroadcastRef.current
            ) {
              lastPlankBroadcastRef.current = plankHoldMsRef.current;
              setPlankHoldMs(Math.round(plankHoldMsRef.current));
            }
          }
          plankLastTimestampRef.current = timestamp;
        }

        return;
      }

      plankLowFramesRef.current += 1;

      if (bodyAngle >= PLANK_MIN_ANGLE) {
        if (plankState !== "adjust") {
          setPlankState("adjust");
        }
        if (plankLowFramesRef.current >= PLANK_WARN_FRAMES) {
          setFeedbackMessage("Press the floor away and lift hips in line with shoulders.", {
            immediate: plankLowFramesRef.current >= PLANK_STRICT_FRAMES,
          });
        }
        return;
      }

      if (plankState !== "adjust") {
        setPlankState("adjust");
      }
      if (plankLowFramesRef.current >= PLANK_WARN_FRAMES) {
        setFeedbackMessage("Reset the plank - squeeze glutes and stack ears over shoulders.", {
          immediate: true,
        });
      }

      if (plankHoldActiveRef.current && plankLowFramesRef.current >= PLANK_STRICT_FRAMES) {
        plankHoldActiveRef.current = false;
        plankLastTimestampRef.current = null;
        plankHoldMsRef.current = 0;
        lastPlankBroadcastRef.current = 0;
        setPlankHoldMs(0);
        setPlankState("setup");
      }
    },
    [plankState, setFeedbackMessage]
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

    if (canvasElement) {
      const context = canvasElement.getContext("2d");
      if (context) {
        context.clearRect(0, 0, canvasElement.width, canvasElement.height);
      }
    }

    lastVideoTimeRef.current = -1;
    clearPoseRefs();
    setIsStarted(false);
    setIsVideoReady(false);
    setFeedbackMessage("Session paused. Hit start when you want the coach back in.", {
      silent: true,
    });
  }, [canvasRef, videoRef, clearPoseRefs, setFeedbackMessage]);

  const start = useCallback(async () => {
    if (!isSupportedMode(mode)) {
      stop();
      setAppError(null);
      setFeedbackMessage("Select an exercise to run live MediaPipe tracking.", { immediate: true });
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
      clearPoseRefs();
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
        mode === "plank"
          ? "Tracking plank alignment - reach long, pack shoulders, and breathe."
          : "Tracking form - move with tempo and stay controlled.",
        { silent: true }
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
      clearPoseRefs();
      setFeedbackMessage("Camera unavailable. Check permissions and try again.");
    }
  }, [mode, setFeedbackMessage, videoRef, clearPoseRefs, stop]);

  useEffect(() => {
    if (!isSupportedMode(mode)) {
      clearPoseRefs();
      setFeedback("MediaPipe tracking is available for the selected exercises.");
      setAppError(null);
      return;
    }

    clearPoseRefs();
    setAppError(null);
    setFeedback(
      mode === "plank"
        ? "Coach ready: press the floor away, lengthen through the crown, and lock the core."
        : "Coach ready: square up to the camera and move with intent."
    );
  }, [mode, clearPoseRefs]);

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

        const timestamp = performance.now();
        const results: PoseLandmarkerResult = landmarkerRef.current.detectForVideo(videoElement, timestamp);
        const poseLandmarks = results.landmarks?.[0];

        if (poseLandmarks && poseLandmarks.length) {
          drawingUtils.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
            lineWidth: 3,
            color: "rgba(147, 51, 234, 0.7)",
          });
          drawingUtils.drawLandmarks(poseLandmarks, { radius: 3, color: "#38bdf8" });

          switch (mode) {
            case "squat":
              analyzeSquat(poseLandmarks);
              break;
            case "pushup":
              analyzePushup(poseLandmarks);
              break;
            case "pullup":
              analyzePullup(poseLandmarks);
              break;
            case "jumpingJack":
              analyzeJumpingJack(poseLandmarks);
              break;
            case "plank":
              analyzePlank(poseLandmarks, timestamp);
              break;
            default:
              break;
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
  }, [isStarted, isVideoReady, mode, analyzeSquat, analyzePushup, analyzePullup, analyzeJumpingJack, analyzePlank, videoRef, canvasRef]);

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
    switch (mode) {
      case "squat":
        if (!isStarted) return "SET";
        return squatState === "down" ? "LOW" : "UP";
      case "pushup":
        if (!isStarted) return "SET";
        return { setup: "SET", lowering: "LOW", press: "TOP" }[pushupState];
      case "pullup":
        if (!isStarted) return "SET";
        return { hang: "HANG", pull: "PULL", top: "TOP" }[pullupState];
      case "jumpingJack":
        if (!isStarted) return "SET";
        return { center: "CENTER", wide: "WIDE" }[jackState];
      case "plank":
        if (!isStarted) return "SET";
        return { setup: "SET", hold: "HOLD", adjust: "FORM" }[plankState];
      default:
        return "READY";
    }
  })();

  return {
    isModeSupported: isSupportedMode(mode),
    isModelReady,
    isStarted,
    isVideoReady,
    repCount,
    plankHoldMs,
    feedback,
    stateLabel,
    appError,
    start,
    stop,
  };
}

