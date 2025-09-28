import { useEffect, useRef, useState, useCallback, type ChangeEvent } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import "./App.css";
import { Volume2 } from "lucide-react";
import plankDemo from "./assets/Alien_Plank_Exercise_Animation_Generated.mp4";
import squatDemo from "./assets/Cartoon_Squat_Animation_Generation.mp4";
import pushupDemo from "./assets/Alien_Push_Up_Animation_Generated.mp4";
import pullupDemo from "./assets/Alien_Pull_Up_Animation_Generation.mp4";
import jumpingjackDemo from "./assets/alien-jumpingjack.mp4";
import alienLogo from "./assets/gemini-logo.png";
import { ReactTyped } from "react-typed";
import { GoogleGenerativeAI } from "@google/generative-ai";

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
const genAI = API_KEY ? new GoogleGenerativeAI(API_KEY) : null;


type ExerciseMode = "squat" | "plank" | "pushup" | "pullup" | "jumpingJack";
type SquatPhase = "start" | "up" | "down";
type PushupPhase = "setup" | "lowering" | "press";
type PullupPhase = "hang" | "pull" | "top";
type JackPhase = "center" | "wide";

type ExerciseConfig = {
  id: ExerciseMode;
  label: string;
  title: string;
  description: string;
  cues: string[];
  stageLabel: string;
  standbyText: string;
  activeText: string;
};

// Put near the top of App.tsx
type Connection = { start: number; end: number };

const POSE_CONNECTIONS: Connection[] = [
  { start: 11, end: 13 }, { start: 13, end: 15 }, // left arm
  { start: 12, end: 14 }, { start: 14, end: 16 }, // right arm
  { start: 11, end: 12 },                         // shoulders
  { start: 23, end: 24 },                         // hips
  { start: 11, end: 23 }, { start: 12, end: 24 }, // torso
  { start: 23, end: 25 }, { start: 25, end: 27 }, // left leg
  { start: 24, end: 26 }, { start: 26, end: 28 }, // right leg
  { start: 27, end: 29 }, { start: 29, end: 31 }, // left foot
  { start: 28, end: 30 }, { start: 30, end: 32 }, // right foot
];

const MODE_OPTIONS: ExerciseConfig[] = [
  {
    id: "squat",
    label: "Squat",
    title: "Master every squat rep.",
    description:
      "Track depth and tempo so your hips, knees, and torso stay aligned rep after rep.",
    cues: [
      "Sit back until thighs land roughly parallel to the ground.",
      "Drive through mid-foot while keeping your chest proud.",
      "Stand tall at lockout and squeeze glutes to finish the rep.",
    ],
    stageLabel: "Squat session",
    standbyText: "Square up to the camera and hit Start when you're ready to squat.",
    activeText: "Maintain an upright torso - depth and tempo are under review.",
  },
  {
    id: "plank",
    label: "Plank",
    title: "Lock in a rock-solid plank.",
    description: "We monitor shoulder, hip, and ankle alignment so your core never sags.",
    cues: [
      "Brace your core and keep hips level with shoulders.",
      "Press the floor away to stay active through the upper back.",
      "Breathe steadily - avoid holding your breath during the hold.",
    ],
    stageLabel: "Plank hold",
    standbyText: "Line up side-on to the camera, then start your plank hold.",
    activeText: "Keep hips, shoulders, and ankles aligned - we'll alert you if they drift.",
  },
  {
    id: "pushup",
    label: "Push-Up",
    title: "Dial in push-up mechanics.",
    description: "Elbow angles and body line are tracked so every rep builds powerful pressing strength.",
    cues: [
      "Lower until elbows reach roughly 90 degrees.",
      "Keep a straight line from head to heel throughout the set.",
      "Lock out without letting elbows flare wide.",
    ],
    stageLabel: "Push-up session",
    standbyText: "Set up side-on so the camera can see your full push-up range.",
    activeText: "Control the descent and drive back to a strong lockout.",
  },
  {
    id: "pullup",
    label: "Pull-Up",
    title: "Own your vertical pull.",
    description: "We watch your hang, path, and chin position to keep every pull-up honest.",
    cues: [
      "Start from a full dead hang with shoulders engaged.",
      "Lead with your chest as you drive elbows down.",
      "Pause briefly with chin above the bar before lowering.",
    ],
    stageLabel: "Pull-up session",
    standbyText: "Face the camera and make sure your pull-up station is fully visible.",
    activeText: "Drive elbows down and lift the chest - we're checking range of motion.",
  },
  {
    id: "jumpingJack",
    label: "Jumping Jack",
    title: "Light up your conditioning.",
    description: "Arm reach and foot spacing are tracked to keep your cardio warm-up crisp.",
    cues: [
      "Snap to a tall, narrow stance between reps.",
      "Reach arms overhead until biceps frame the ears.",
      "Land softly with knees slightly bent on every return.",
    ],
    stageLabel: "Jumping jack session",
    standbyText: "Stand centered in frame so full arm span stays visible.",
    activeText: "Hit full extension on each rep - we're monitoring symmetry and rhythm.",
  },
];

const DEMO_VIDEOS: Record<ExerciseMode, string> = {
  squat: squatDemo,
  plank: plankDemo,
  pushup: pushupDemo,
  pullup: pullupDemo,
  jumpingJack: jumpingjackDemo,
};

const EXERCISE_CONFIG = MODE_OPTIONS.reduce(
  (acc, cfg) => {
    acc[cfg.id] = cfg;
    return acc;
  },
  {} as Record<ExerciseMode, ExerciseConfig>
);

async function fetchExerciseLines(mode: ExerciseMode): Promise<string[]> {
  const fallback = [
    "Master every squat rep.",
    "Lock in a rock-solid plank.",
    "Dial in push-up mechanics.",
  ];

  if (!genAI) return fallback;

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const prompt =
      `Generate 6 short, punchy coaching lines for ${mode}.\n` +
      `Under 8 words each. Energetic, clear.\n` +
      `No numbering or bullets. One line per line.`;

    const result = await model.generateContent(prompt);
    const t = result.response.text(); // <-- define `t`, not `text`

    const lines = t
      .split(/\r?\n/)
      .map((line: string) => line.replace(/^[\s\-\*\d\.\)]+/, "").trim())
      .filter((line: string) => line.length > 0)
      .slice(0, 6);

    return lines.length ? lines : fallback;
  } catch (err) {
    console.error("Gemini request failed:", err);
    return fallback;
  }
}



export default function App() {
  const [typedLines, setTypedLines] = useState<string[]>([]);
  const [linesLoading, setLinesLoading] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);

  const [mode, setMode] = useState<ExerciseMode>("squat");
  const [isStarted, setIsStarted] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [squatState, setSquatState] = useState<SquatPhase>("start");
  const [pushupState, setPushupState] = useState<PushupPhase>("setup");
  const [pullupState, setPullupState] = useState<PullupPhase>("hang");
  const [jackState, setJackState] = useState<JackPhase>("center");
  const [feedback, setFeedback] = useState("");
  const [isDemoPlaying, setIsDemoPlaying] = useState(false);
  const [demoKey, setDemoKey] = useState(0);

  const lastSpeechTime = useRef(Date.now());
  const speak = useCallback((t: string) => {
    const now = Date.now();
    if (now - lastSpeechTime.current > 3000) {
      speechSynthesis.speak(new SpeechSynthesisUtterance(t));
      lastSpeechTime.current = now;
    }
  }, []);

  const currentConfig = EXERCISE_CONFIG[mode];

  useEffect(() => {
    setRepCount(0);
    setFeedback("");
    setSquatState("start");
    setPushupState("setup");
    setPullupState("hang");
    setJackState("center");
    setIsDemoPlaying(false);
    setDemoKey((k) => k + 1);
  }, [mode]);

  // --- Initialize MediaPipe
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await tf.setBackend("webgl");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      if (cancelled) return;
      landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- Camera start
  const handleStart = async () => {
    if (isDemoPlaying) setIsDemoPlaying(false);
    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    video.srcObject = stream;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Video error"));
    });

    await video.play();
    setIsVideoReady(true);
    setIsStarted(true);
    setFeedback("");
  };
  const handleStop = () => {
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      v.srcObject = null;
    }
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
    setIsStarted(false);
    setIsVideoReady(false);
    setFeedback("");
    setIsDemoPlaying(false);
  };

  const handleModeChange = (event: ChangeEvent<HTMLSelectElement>) => {
    setMode(event.target.value as ExerciseMode);
  };

  const handleShowDemo = () => {
    if (isStarted) {
      handleStop();
    }
    setIsDemoPlaying(true);
    setDemoKey((k) => k + 1);
    setFeedback("");
  };

  const handleCloseDemo = () => {
    setIsDemoPlaying(false);
  };


  // --- Helpers
  const angle = (a: number[], b: number[], c: number[]) => {
    const r =
      Math.atan2(c[1] - b[1], c[0] - b[0]) -
      Math.atan2(a[1] - b[1], a[0] - b[0]);
    let d = Math.abs((r * 180) / Math.PI);
    if (d > 180) d = 360 - d;
    return d;
  };

  const distance = (a: number[], b: number[]) => {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  };

  const analyzeSquat = useCallback(
    (lm: any[]) => {
      const hip = [lm[24].x, lm[24].y];
      const knee = [lm[26].x, lm[26].y];
      const ankle = [lm[28].x, lm[28].y];
      const kneeAngle = angle(hip, knee, ankle);

      if (kneeAngle >= 170) {
        if (squatState === "down") {
          setRepCount((p) => {
            const n = p + 1;
            speak(`Rep ${n}`);
            return n;
          });
        }
        setSquatState("up");
      } else if (kneeAngle <= 100 && squatState === "up") {
        setSquatState("down");
        if (kneeAngle < 70) {
          setFeedback("Too low! Aim for about 90°");
          speak("Too low");
        }
      }
    },
    [squatState, speak]
  );

  const analyzePlank = useCallback(
    (lm: any[]) => {
      const shoulder = [lm[12].x, lm[12].y];
      const hip = [lm[24].x, lm[24].y];
      const ankle = [lm[28].x, lm[28].y];
      const body = angle(shoulder, hip, ankle);
      if (body < 160) {
        setFeedback("Keep your body straight!");
        speak("Straighten your body");
      } else {
        setFeedback("Good form!");
      }
    },
    [speak]
  );

  const analyzePushup = useCallback(
    (lm: any[]) => {
      const shoulder = [lm[12].x, lm[12].y];
      const elbow = [lm[14].x, lm[14].y];
      const wrist = [lm[16].x, lm[16].y];
      const elbowAngle = angle(shoulder, elbow, wrist);

      if (elbowAngle >= 160) {
        if (pushupState === "lowering") {
          setRepCount((p) => {
            const n = p + 1;
            speak(`Rep ${n}`);
            return n;
          });
        }
        if (pushupState !== "press") {
          setPushupState("press");
          setFeedback("Strong lockout - brace your core.");
        }
      } else if (elbowAngle <= 90) {
        if (pushupState !== "lowering") {
          setPushupState("lowering");
          setFeedback("Control the descent until elbows hit 90°.");
        }
      } else if (pushupState !== "setup") {
        setPushupState("setup");
      }
    },
    [pushupState, speak]
  );

  const analyzePullup = useCallback(
    (lm: any[]) => {
      const avgWristY = (lm[15].y + lm[16].y) / 2;
      const avgShoulderY = (lm[11].y + lm[12].y) / 2;
      const noseY = lm[0].y;

      if (noseY <= avgShoulderY + 0.02) {
        if (pullupState === "pull") {
          setRepCount((p) => {
            const n = p + 1;
            speak(`Rep ${n}`);
            return n;
          });
        }
        if (pullupState !== "top") {
          setPullupState("top");
          setFeedback("Hold the top - drive elbows back.");
        }
      } else if (avgWristY - avgShoulderY > 0.18) {
        if (pullupState !== "hang") {
          setPullupState("hang");
          setFeedback("Full hang - engage lats before the next pull.");
        }
      } else if (pullupState !== "pull") {
        setPullupState("pull");
        setFeedback("Lead with your chest and squeeze shoulder blades.");
      }
    },
    [pullupState, speak]
  );

  const analyzeJumpingJack = useCallback(
    (lm: any[]) => {
      const leftAnkle = [lm[27].x, lm[27].y];
      const rightAnkle = [lm[28].x, lm[28].y];
      const leftWrist = [lm[15].x, lm[15].y];
      const rightWrist = [lm[16].x, lm[16].y];
      const ankleGap = distance(leftAnkle, rightAnkle);
      const wristGap = distance(leftWrist, rightWrist);
      const shoulderY = (lm[11].y + lm[12].y) / 2;
      const wristY = (leftWrist[1] + rightWrist[1]) / 2;

      if (ankleGap > 0.5 && wristGap > 0.35 && wristY < shoulderY) {
        if (jackState === "center") {
          setRepCount((p) => {
            const n = p + 1;
            speak(`Rep ${n}`);
            return n;
          });
        }
        if (jackState !== "wide") {
          setJackState("wide");
          setFeedback("Big extension - keep arms tall overhead.");
        }
      } else if (ankleGap < 0.3 && wristGap < 0.28) {
        if (jackState !== "center") {
          setJackState("center");
          setFeedback("Snap back to center and brace your core.");
        }
      }
    },
    [jackState, speak]
  );

  // --- Frame loop
  useEffect(() => {
    if (isDemoPlaying) return;
    if (
      !isStarted ||
      !isVideoReady ||
      !videoRef.current ||
      !canvasRef.current ||
      !landmarkerRef.current
    )
      return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const draw = new DrawingUtils(ctx);

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let raf = 0;
    const loop = async () => {
      const lm = landmarkerRef.current!;
      const results = await lm.detectForVideo(video, performance.now());

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (results.landmarks?.[0]) {
        const l = results.landmarks[0];
        draw.drawLandmarks(l);
        draw.drawConnectors(l, POSE_CONNECTIONS);

        switch (mode) {
          case "squat":
            analyzeSquat(l);
            break;
          case "plank":
            analyzePlank(l);
            break;
          case "pushup":
            analyzePushup(l);
            break;
          case "pullup":
            analyzePullup(l);
            break;
          case "jumpingJack":
            analyzeJumpingJack(l);
            break;
          default:
            analyzePlank(l);
        }
      }

      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [
    isStarted,
    isVideoReady,
    mode,
    analyzePlank,
    analyzeSquat,
    analyzePushup,
    analyzePullup,
    analyzeJumpingJack,
    isDemoPlaying,
  ]);

  // --- Cleanup
  useEffect(() => {
    return () => {
      const v = videoRef.current;
      if (v?.srcObject) {
        (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const stateDisplay = (() => {
    if (isDemoPlaying) return "DEMO";
    switch (mode) {
      case "squat":
        return { start: "SET", up: "UP", down: "LOW" }[squatState];
      case "pushup":
        return { setup: "SET", lowering: "LOW", press: "TOP" }[pushupState];
      case "pullup":
        return { hang: "HANG", pull: "PULL", top: "TOP" }[pullupState];
      case "jumpingJack":
        return { center: "CENTER", wide: "WIDE" }[jackState];
      case "plank":
        return isStarted ? "HOLD" : "SET";
      default:
        return "READY";
    }
  })();

useEffect(() => {
  let alive = true;
  (async () => {
    try {
      setLinesLoading(true);
      const lines = await fetchExerciseLines(mode);
      if (alive) setTypedLines(lines);
    } catch (e) {
      console.error(e);
      if (alive) {
        setTypedLines([
          "Master every squat rep.",
          "Lock in a rock-solid plank.",
          "Dial in push-up mechanics.",
        ]);
      }
    } finally {
      if (alive) setLinesLoading(false);
    }
  })();
  return () => { alive = false; };
}, [mode]);

  return (
    <div className="app-background">
      <div className="app-ribbon">owl hawks 2025</div>
      <div className="app-frame">
        <header className="app-header">
          <div className="brand">
            <img
              src={alienLogo}
              alt="Gemini Logo"
              className="brand-logo"
            />
            <div>
              <p className="brand-title">AI Form Coach</p>
              <p className="brand-subtitle">Real-time MediaPipe biomechanics</p>
            </div>
          </div>
          <div className="header-tools">
            <span className="mode-chip">{currentConfig.label}</span>
            <div className="header-status" aria-live="polite">
              <span className={`status-dot ${isStarted ? "is-live" : ""}`} />
              <span className="status-text">{isStarted ? "Camera live" : "Standby"}</span>
            </div>
          </div>
        </header>

        <main className="content-grid">
          <section className="info-panel">
           <h1 className="section-title">
            {typedLines.length ? (
              <ReactTyped
                key={typedLines.join("|")}
                strings={typedLines}
                typeSpeed={50}
                backSpeed={30}
                backDelay={1200}
                loop
              />
            ) : (
              currentConfig.title
            )}
          </h1>
            <p className="section-copy">{currentConfig.description}</p>

            <div className="mode-control">
              <label htmlFor="exercise-select" className="mode-label">
                Exercise mode
              </label>
              <div className="mode-switch">
                <select
                  id="exercise-select"
                  className="mode-select"
                  value={mode}
                  onChange={handleModeChange}
                >
                  {MODE_OPTIONS.map(({ id, label }) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className={`demo-button ${isDemoPlaying ? "is-active" : ""}`}
                  onClick={isDemoPlaying ? handleCloseDemo : handleShowDemo}
                >
                  {isDemoPlaying ? "Close demo" : "Show me how"}
                </button>
              </div>
            </div>

            <div className="stat-card" aria-live="polite">
              <div className="stat">
                <span className="stat-label">Rep counter</span>
                <span className="stat-value">{isDemoPlaying || mode === "plank" ? "--" : repCount}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Current state</span>
                <span className="stat-value">{stateDisplay}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Audio cues</span>
                <span className="stat-icon" title="Voice feedback enabled">
                  <Volume2 size={18} />
                </span>
              </div>
            </div>

            <div className="note-card">
              <h2>Form cues for {currentConfig.label}</h2>
              <ul>
                {currentConfig.cues.map((cue) => (
                  <li key={cue}>{cue}</li>
                ))}
              </ul>
            </div>
          </section>

          <section className="stage-panel">
            <div className="video-stage">
              <div className="video-stage__header">
                <span className="video-stage__title">{currentConfig.stageLabel}</span>
                <span className="video-stage__chip">{isDemoPlaying ? "Demo" : isStarted ? "Analyzing" : "Ready"}</span>
              </div>
              <div className="video-wrapper">
                <video
                  ref={videoRef}
                  className={`live-feed ${isDemoPlaying ? "is-hidden" : ""}`}
                  playsInline
                  muted
                />
                <canvas
                  ref={canvasRef}
                  className={`live-feed ${isDemoPlaying ? "is-hidden" : ""}`}
                />
                {isDemoPlaying && (
                  <video
                    key={`${mode}-${demoKey}`}
                    className="demo-feed"
                    src={DEMO_VIDEOS[mode]}
                    autoPlay
                    loop
                    muted
                    playsInline
                    controls
                  />
                )}
              </div>
              <div className="stage-actions">
                {isDemoPlaying ? (
                  <button type="button" className="cta-button is-stop" onClick={handleCloseDemo}>
                    Close demo
                  </button>
                ) : !isStarted ? (
                  <button type="button" className="cta-button is-start" onClick={handleStart}>
                    Start camera
                  </button>
                ) : (
                  <button type="button" className="cta-button is-stop" onClick={handleStop}>
                    Stop camera
                  </button>
                )}
              </div>
            </div>

            <div className={`feedback-banner ${feedback || isDemoPlaying ? "has-text" : ""}`}>
              {isDemoPlaying
                ? "Watch the demo, then start the camera when you're ready."
                : feedback
                ? feedback
                : isStarted
                ? currentConfig.activeText
                : currentConfig.standbyText}
            </div>
          </section>
        </main>

        <footer className="app-footer">
          Built for the Next Frontier Health track with React, TensorFlow.js, and MediaPipe
        </footer>
      </div>
    </div>
  );

}
