import { useEffect, useRef, useState, useCallback } from "react";
import {
  Activity,
  Clock3,
  Sparkles,
  Volume2,
  Play,
  Square,
  Camera,
  Zap,
  Target,
  Flame,
} from "lucide-react";
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import "./App.css";
import "./utilities.css";

type ExerciseMode = "squat" | "plank";
type SquatPhase = "start" | "up" | "down";
type LandmarkList = ReadonlyArray<NormalizedLandmark>;
type Point2D = readonly [number, number];

type SpeakOptions = { immediate?: boolean };
type FeedbackOptions = {
  immediate?: boolean;
  allowRepeat?: boolean;
  silent?: boolean;
};

const HIGHLIGHTS = [
  {
    title: "Movement IQ",
    description: "33-point pose detection keeps your joints stacked and braced under load.",
    Icon: Activity,
  },
  {
    title: "Tempo Precision",
    description: "Live pace tracking balances drive and control for every strength block.",
    Icon: Clock3,
  },
  {
    title: "Adaptive Coaching",
    description: "Voice cues respond to each phase so every rep lands with intent.",
    Icon: Sparkles,
  },
];

const PROGRAMS = [
  {
    title: "Athletic Strength",
    description: "Dial in heavy lifts with rep-by-rep joint tracking and tempo cues.",
    focus: "Barbell and kettlebell progressions",
    Icon: Target,
  },
  {
    title: "Engine Conditioning",
    description: "Rotate through high-output intervals while AI protects your mechanics.",
    focus: "Sprint, row, and plyometric stacks",
    Icon: Zap,
  },
  {
    title: "Mobility Lab",
    description: "Unlock range with dynamic sequencing and stability checkpoints.",
    focus: "Flow sessions and pillar prep",
    Icon: Flame,
  },
];

const SUCCESS_STORIES = [
  {
    quote: "Pulse Coach cleaned up my squat depth in two weeks. The live cues keep me honest every set.",
    name: "Janelle R.",
    role: "Amateur powerlifter",
    stat: "+35 lb back squat",
  },
  {
    quote: "I finally stay consistent with accessory work because the program shows me exactly what to fix.",
    name: "Marcus L.",
    role: "Semi-pro hooper",
    stat: "6% body fat drop",
  },
  {
    quote: "The plank lab rebuilt my core after a long layoff. It is like having a coach in the room.",
    name: "Akira T.",
    role: "Busy founder",
    stat: "Pain-free again",
  },
];

const MEMBERSHIP_PLANS = [
  {
    name: "Studio Access",
    price: "39",
    period: "per month",
    description: "For self-guided athletes who want visual feedback without the fluff.",
    features: [
      "Unlimited AI lab sessions",
      "Real-time rep counter",
      "Movement quality snapshots",
    ],
  },
  {
    name: "Coached Upgrade",
    price: "79",
    period: "per month",
    description: "Weekly programming drops plus form reviews from our coaching staff.",
    features: [
      "Everything in Studio Access",
      "Tailored weekly blocks",
      "Monthly technique audit",
      "Priority chat support",
    ],
    recommended: true,
  },
  {
    name: "Performance Team",
    price: "129",
    period: "per month",
    description: "For competitors chasing the podium with data-backed refinement.",
    features: [
      "All coached features",
      "Live feedback sessions",
      "Custom recovery protocols",
      "Quarterly testing days",
    ],
  },
];

const FAQ_ITEMS = [
  {
    question: "Do I need special equipment?",
    answer: "All you need is a laptop or tablet with a camera. A tripod or stable surface gives the best angles.",
  },
  {
    question: "Will this replace my current coach?",
    answer: "Pulse Coach acts like a second set of eyes. Many athletes pair it with an in-person or remote coach.",
  },
  {
    question: "How private is my session footage?",
    answer: "Video stays on your device during AI analysis. Nothing is uploaded unless you choose to share a review.",
  },
  {
    question: "Can beginners use the platform?",
    answer: "Absolutely. Mode guides keep cues simple and the movement library scales from foundations to elite work.",
  },
];

const SESSION_CUES: Record<ExerciseMode, string[]> = {
  squat: [
    "Drive knees outward to stack directly over your toes.",
    "Sit hips back, keep your brace tall, and lock in your lats.",
    "Hit parallel, then explode up through the floor.",
  ],
  plank: [
    "Press the ground away and create space between shoulder blades.",
    "Draw ribs toward hips to lock in the midline.",
    "Squeeze glutes so ankles, hips, and shoulders stay aligned.",
  ],
};

const MODE_TITLES: Record<ExerciseMode, string> = {
  squat: "Squat Mechanics",
  plank: "Core Alignment",
};

const MODE_TAGLINES: Record<ExerciseMode, string> = {
  squat: "Build powerful depth and stability rep after rep.",
  plank: "Hold pristine lines with total core engagement.",
};

const SPEECH_COOLDOWN_MS = 3000;

const STANDING_KNEE_ANGLE = 165;
const BOTTOM_KNEE_ANGLE = 100;
const MIN_HIP_DELTA_FOR_REP = 0.07;
const MIN_KNEE_WIDTH_RATIO = 0.75;
const HIP_RETURN_TOLERANCE = 0.018;
const HIP_BELOW_KNEE_THRESHOLD = 0.02;
const FRONT_KNEE_ANGLE_MAX = 125;

const VISION_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

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

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const lastSpeechTime = useRef<number>(0);
  const standingHipRef = useRef<number | null>(null);
  const bottomHipRef = useRef<number | null>(null);
  const standingHipWidthRef = useRef<number | null>(null);

  const [mode, setMode] = useState<ExerciseMode>("squat");
  const [isStarted, setIsStarted] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [squatState, setSquatState] = useState<SquatPhase>("start");
  const [feedback, setFeedback] = useState("Coach ready. Press start to begin.");
  const [appError, setAppError] = useState<string | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);

  const speak = useCallback((text: string, options: SpeakOptions = {}) => {
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
  }, []);

  const setFeedbackMessage = useCallback(
    (message: string, options: FeedbackOptions = {}) => {
      setFeedback((prev) => {
        const changed = prev !== message;
        if (changed) {
          if (!options.silent) {
            speak(message, { immediate: options.immediate });
          }
          return message;
        }

        if (!options.silent && options.allowRepeat) {
          speak(message, { immediate: options.immediate });
        }

        return prev;
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
        baseHipWidth && baseHipWidth > 0 && kneeWidth !== null
          ? kneeWidth / baseHipWidth
          : null;
      const kneesWideEnough =
        kneeWidthRatio !== null && kneeWidthRatio >= MIN_KNEE_WIDTH_RATIO;
      const kneesWideBottom =
        kneesWideEnough && kneeAngle !== null && kneeAngle <= FRONT_KNEE_ANGLE_MAX;

      if (standingHipRef.current === null) {
        standingHipRef.current = avgHipY;
        if (hipWidth !== null && hipWidth > 0 && standingHipWidthRef.current === null) {
          standingHipWidthRef.current = hipWidth;
        }
      }

      const depthFromTop =
        standingHipRef.current !== null ? avgHipY - standingHipRef.current : 0;

      const kneeAngleIndicatesStanding = kneeAngle !== null && kneeAngle >= STANDING_KNEE_ANGLE;
      const hipReturn =
        Math.abs(depthFromTop) <= Math.max(HIP_RETURN_TOLERANCE, MIN_HIP_DELTA_FOR_REP * 0.35);
      const hipPositionIndicatesStanding = hipReturn || hipVerticalDiff <= 0;
      const isStanding = kneeAngleIndicatesStanding && hipPositionIndicatesStanding;

      const kneeAngleIndicatesBottom = kneeAngle !== null && kneeAngle <= BOTTOM_KNEE_ANGLE;
      const depthSufficient = depthFromTop >= MIN_HIP_DELTA_FOR_REP;
      const hipBelowKneeStrong = hipVerticalDiff >= HIP_BELOW_KNEE_THRESHOLD;
      const bottomDetected =
        depthSufficient && (kneeAngleIndicatesBottom || hipBelowKneeStrong || kneesWideBottom);

      if (isStanding) {
        if (
          squatState === "down" &&
          bottomHipRef.current !== null &&
          standingHipRef.current !== null &&
          bottomHipRef.current - standingHipRef.current >= MIN_HIP_DELTA_FOR_REP
        ) {
          setRepCount((prev) => {
            const next = prev + 1;
            speak(`Rep ${next}`, { immediate: true });
            return next;
          });
        }

        bottomHipRef.current = null;

        if (hipWidth !== null && hipWidth > 0) {
          standingHipWidthRef.current =
            standingHipWidthRef.current === null
              ? hipWidth
              : Math.min(standingHipWidthRef.current, hipWidth);
        }

        if (squatState !== "up") {
          standingHipRef.current = avgHipY;
          setSquatState("up");
          setFeedbackMessage("Stand tall, reset your brace, and own the next drive.");
        } else if (standingHipRef.current !== null) {
          standingHipRef.current = Math.min(standingHipRef.current, avgHipY);
        }
        return;
      }

      if (squatState === "up" && bottomDetected) {
        bottomHipRef.current = avgHipY;
        setSquatState("down");
        if (kneeAngle !== null && kneeAngle < 82) {
          setFeedbackMessage("Hold parallel - stop slightly higher to protect your knees.", {
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
    [setFeedbackMessage, speak]
  );

  useEffect(() => {
    let isCancelled = false;

    const initializeModel = async () => {
      try {
        setAppError(null);
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

  useEffect(() => {
    setRepCount(0);
    setSquatState("start");
    standingHipRef.current = null;
    bottomHipRef.current = null;
    standingHipWidthRef.current = null;
    if (!isStarted) {
      setFeedback(
        mode === "squat"
          ? "Coach ready: brace tall, knees track over toes, and sit back confidently."
          : "Coach ready: press the floor away, lengthen through the crown, and lock the core."
      );
    }
  }, [isStarted, mode]);

  const handleStart = async () => {
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
  };

  const handleStop = () => {
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
  };

  useEffect(() => {
    if (!isStarted || !isVideoReady) {
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
  }, [isStarted, isVideoReady, mode, analyzePlank, analyzeSquat]);

  useEffect(() => {
    const videoElement = videoRef.current;
    return () => {
      if (videoElement) {
        stopMediaStream(videoElement.srcObject as MediaStream | null);
        videoElement.srcObject = null;
      }
    };
  }, []);

  const sessionCues = SESSION_CUES[mode];
  const heroCue = sessionCues[0] ?? "Move with control.";
  const feedbackDisplay = feedback || (isStarted ? "Analyzing..." : "Coach ready. Press start to begin.");
  const activeModeTitle = MODE_TITLES[mode];
  const modeTagline = MODE_TAGLINES[mode];
  const sessionStatus = !isModelReady
    ? "Loading pose intelligence"
    : isStarted
    ? "Tracking live movement"
    : "Coach ready";
  const stateDescriptor = mode === "squat"
    ? squatState === "start"
      ? "Dial in your setup"
      : squatState === "up"
      ? "Drive tall"
      : "Control the drop"
    : "Hold strong";
  const voiceStatus = isStarted
    ? "Streaming personalized cues"
    : isModelReady
    ? "Standing by"
    : "Warming up";

  return (
    <div className="site-wrapper">
      <div className="site-gradient" />
      <div className="site-inner">
        <header className="site-header">
          <div className="brand">
            <div className="brand__mark">PC</div>
            <div>
              <p className="brand__name">Pulse Coach</p>
              <p className="brand__tagline">Human performance engineered</p>
            </div>
          </div>
          <nav className="site-nav">
            <a href="#programs">Programs</a>
            <a href="#studio">Vision Studio</a>
            <a href="#stories">Stories</a>
            <a href="#plans">Plans</a>
          </nav>
          <a className="btn btn-secondary" href="#studio">
            Book a strategy call
          </a>
        </header>

        <main className="site-main">
          <section className="hero" id="home">
            <div className="hero__content">
              <span className="hero__eyebrow">
                <Sparkles size={16} />
                Hybrid training plus vision AI
              </span>
              <h1 className="hero__title">
                Own every rep with <span>Pulse Coach</span>
              </h1>
              <p className="hero__description">
                Elite-level coaching that runs through your camera. Switch between squat and plank labs, capture every joint in real time, and never guess about form again.
              </p>
              <div className="hero__actions">
                <a className="btn btn-primary" href="#studio">
                  Launch the vision lab
                </a>
                <a className="btn btn-text" href="#plans">
                  Explore memberships
                </a>
              </div>
              <div className="hero__metrics">
                <div className="stat-card">
                  <p className="stat-card__label">Live rep counter</p>
                  <p className="stat-card__value">{repCount}</p>
                  <span className="stat-card__caption">Session total</span>
                </div>
                <div className="stat-card">
                  <p className="stat-card__label">Mode focus</p>
                  <p className="stat-card__value">{activeModeTitle}</p>
                  <span className="stat-card__caption">{stateDescriptor}</span>
                </div>
                <div className="stat-card">
                  <p className="stat-card__label">Voice coaching</p>
                  <p className="stat-card__value">{voiceStatus}</p>
                  <span className="stat-card__caption">Auto cues every rep</span>
                </div>
              </div>
            </div>

            <div className="hero__visual">
              <div className="hero-panel">
                <span className="chip chip-soft">Today&apos;s focus</span>
                <h3>{activeModeTitle}</h3>
                <p>{modeTagline}</p>
                <div className="hero-panel__stats">
                  <div>
                    <p className="hero-panel__label">Session status</p>
                    <p className="hero-panel__value">{sessionStatus}</p>
                  </div>
                  <div>
                    <p className="hero-panel__label">Next cue</p>
                    <p className="hero-panel__value">{heroCue}</p>
                  </div>
                </div>
                <div className="hero-panel__cta">
                  <Volume2 size={18} />
                  <span>{voiceStatus}</span>
                </div>
              </div>
            </div>
          </section>

          <section className="programs" id="programs">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Activity size={16} />
                Training tracks
              </span>
              <div>
                <h2>Programs built for athletes who want more</h2>
                <p>Choose a track and stack it with lab sessions that sharpen every pattern.</p>
              </div>
            </div>
            <div className="program-grid">
              {PROGRAMS.map(({ title, description, focus, Icon }) => (
                <div key={title} className="program-card">
                  <div className="program-card__icon">
                    <Icon size={20} />
                  </div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                  <span>{focus}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="lab" id="studio">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Camera size={16} />
                Vision studio
              </span>
              <div>
                <h2>Train inside the AI coaching lab</h2>
                <p>Pick your focus block and let the camera grade every frame of movement.</p>
              </div>
            </div>

            <div className="lab__grid">
              <div className="lab__panel">
                <div className="lab__status">
                  <span className={`chip ${isStarted ? "chip-live" : ""}`}>{sessionStatus}</span>
                  <span className="chip chip-soft">
                    <Volume2 size={16} />
                    {voiceStatus}
                  </span>
                </div>

                <div className="mode-switch">
                  {(["squat", "plank"] as ExerciseMode[]).map((item) => (
                    <button
                      key={item}
                      className={`mode-button ${mode === item ? "mode-button--active" : ""}`}
                      onClick={() => setMode(item)}
                      disabled={!isModelReady && mode !== item}
                    >
                      {MODE_TITLES[item]}
                    </button>
                  ))}
                </div>

                <p className="lab__tagline">{MODE_TAGLINES[mode]}</p>

                <div className="lab__feedback">
                  <h3>Coach feedback</h3>
                  <p>{feedbackDisplay}</p>
                </div>

                {appError && <div className="lab__alert lab__alert--error">{appError}</div>}

                {!appError && !isModelReady && (
                  <div className="lab__alert lab__alert--info">
                    <span className="spinner" />
                    Loading pose model...
                  </div>
                )}

                <button
                  onClick={isStarted ? handleStop : handleStart}
                  className={`btn btn-primary lab__action ${isStarted ? "btn-danger" : ""}`}
                  disabled={!isStarted && !isModelReady}
                >
                  {isStarted ? <Square size={18} /> : <Play size={18} />}
                  {isStarted ? "End session" : isModelReady ? "Start session" : "Loading model"}
                </button>

                <div className="lab__cues">
                  <h4>Focus cues</h4>
                  <ul>
                    {sessionCues.map((cue, index) => (
                      <li key={index}>{cue}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="lab__video">
                <div className="video-shell">
                  <div className="video-frame">
                    <video
                      ref={videoRef}
                      playsInline
                      muted
                      className="video-element"
                      style={{ transform: "scaleX(-1)" }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="video-element"
                      style={{ transform: "scaleX(-1)" }}
                    />

                    <div className="video-hud">
                      <div className="hud-counter">
                        <span className="hud-counter__label">Reps</span>
                        <span className="hud-counter__value">{repCount}</span>
                      </div>
                    </div>

                    {!isVideoReady && (
                      <div className="video-placeholder">
                        {isStarted ? (
                          <div>
                            <div className="spinner spinner--lg" />
                            <p>Initializing camera</p>
                            <span>Press end session if you need to reset.</span>
                          </div>
                        ) : (
                          <div>
                            <div className="video-icon">
                              <Camera size={28} />
                            </div>
                            <p>Camera ready</p>
                            <span>Position yourself so we can see ankles to shoulders.</span>
                          </div>
                        )}
                      </div>
                    )}


                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="features" id="technology">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Sparkles size={16} />
                Movement technology
              </span>
              <div>
                <h2>Why athletes trust the lab</h2>
                <p>Every layer of Pulse Coach is built to feel like a dedicated in-person session.</p>
              </div>
            </div>

            <div className="features-grid">
              {HIGHLIGHTS.map(({ title, description, Icon }) => (
                <article key={title} className="feature-card">
                  <div className="feature-card__icon">
                    <Icon size={20} />
                  </div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="stories" id="stories">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Flame size={16} />
                Athlete stories
              </span>
              <div>
                <h2>Results from the community</h2>
                <p>From garage gyms to pro clubs, athletes use Pulse Coach to sharpen the details.</p>
              </div>
            </div>

            <div className="stories-grid">
              {SUCCESS_STORIES.map(({ quote, name, role, stat }) => (
                <blockquote key={name} className="story-card">
                  <p className="story-card__quote">"{quote}"</p>
                  <div className="story-card__meta">
                    <span className="story-card__name">{name}</span>
                    <span className="story-card__role">{role}</span>
                    <span className="story-card__stat">{stat}</span>
                  </div>
                </blockquote>
              ))}
            </div>
          </section>

          <section className="membership" id="plans">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Target size={16} />
                Memberships
              </span>
              <div>
                <h2>Pick your lane and keep building</h2>
                <p>Flexible memberships for lifters, field sport athletes, and teams.</p>
              </div>
            </div>

            <div className="membership-grid">
              {MEMBERSHIP_PLANS.map(({ name, price, period, description, features, recommended }) => (
                <div key={name} className={`plan-card ${recommended ? "plan-card--featured" : ""}`}>
                  {recommended && <span className="chip chip-live plan-card__badge">Most popular</span>}
                  <h3>{name}</h3>
                  <p className="plan-card__price">
                    ${price}
                    <span>{period}</span>
                  </p>
                  <p className="plan-card__description">{description}</p>
                  <ul className="plan-card__features">
                    {features.map((feature) => (
                      <li key={feature}>{feature}</li>
                    ))}
                  </ul>
                  <a className="btn btn-primary plan-card__cta" href="#studio">
                    Start 7 day trial
                  </a>
                </div>
              ))}
            </div>
          </section>

          <section className="faq">
            <div className="section-heading">
              <span className="hero__eyebrow">
                <Sparkles size={16} />
                FAQ
              </span>
              <div>
                <h2>Need clarity before you jump in?</h2>
                <p>We pulled the most common questions from hundreds of athlete onboarding calls.</p>
              </div>
            </div>

            <div className="faq-grid">
              {FAQ_ITEMS.map(({ question, answer }) => (
                <div key={question} className="faq-card">
                  <h3>{question}</h3>
                  <p>{answer}</p>
                </div>
              ))}
            </div>
          </section>
        </main>

        <footer className="site-footer">
          <div className="brand">
            <div className="brand__mark">PC</div>
            <div>
              <p className="brand__name">Pulse Coach</p>
              <p className="brand__tagline">Precision coaching crafted with React and MediaPipe.</p>
            </div>
          </div>
          <div className="footer-links">
            <a href="#programs">Programs</a>
            <a href="#studio">Vision studio</a>
            <a href="#plans">Memberships</a>
            <a href="#stories">Success stories</a>
          </div>
          <p className="footer-note">Train smarter, recover better, and keep the momentum rolling.</p>
        </footer>
      </div>
    </div>
  );
}















