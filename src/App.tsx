import { useEffect, useRef, useState, type ChangeEvent } from "react";
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
import { usePoseCoach, type SupportedMode } from "./mediapipe/usePoseCoach";

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
const genAI = API_KEY ? new GoogleGenerativeAI(API_KEY) : null;

type ExerciseMode = "squat" | "plank" | "pushup" | "pullup" | "jumpingJack";

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

const SUPPORTED_POSE_MODES: readonly SupportedMode[] = ["squat", "plank"];
const isSupportedPoseMode = (mode: ExerciseMode): mode is SupportedMode =>
  (SUPPORTED_POSE_MODES as readonly ExerciseMode[]).includes(mode as SupportedMode);

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
    const t = result.response.text();

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

function formatHoldTime(ms: number): string {
  if (ms <= 0) {
    return "0.0s";
  }
  const totalSeconds = ms / 1000;
  if (totalSeconds >= 60) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return minutes + ":" + seconds.toString().padStart(2, "0");
  }
  return totalSeconds.toFixed(1) + "s";
}

export default function App() {
  const [typedLines, setTypedLines] = useState<string[]>([]);
  const [activePage, setActivePage] = useState<"home" | "about">("home");
  const [mode, setMode] = useState<ExerciseMode>("squat");
  const [isDemoPlaying, setIsDemoPlaying] = useState(false);
  const [demoKey, setDemoKey] = useState(0);
  const [plankHoldMs, setPlankHoldMs] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const plankTimerRef = useRef<number | null>(null);

  const supportedMode = isSupportedPoseMode(mode) ? mode : null;

  const {
    isModeSupported,
    isModelReady,
    isStarted,
    isVideoReady,
    repCount,
    feedback: coachFeedback,
    stateLabel,
    appError,
    start,
    stop,
  } = usePoseCoach({ mode: supportedMode, videoRef, canvasRef });

  const currentConfig = EXERCISE_CONFIG[mode];
  const statusText = isStarted
    ? "Camera live"
    : isModeSupported
    ? "Standby"
    : "Mode unavailable";
  const stageChipLabel = isDemoPlaying
    ? "Demo"
    : isStarted
    ? "Analyzing"
    : isModeSupported
    ? "Ready"
    : "Offline";
  const metricLabel = mode === "plank" ? "Hold time" : "Rep counter";
  const metricValue = isDemoPlaying
    ? "--"
    : mode === "plank"
    ? isModeSupported && isStarted
      ? formatHoldTime(plankHoldMs)
      : "0.0s"
    : String(repCount);
  const stateDisplay = isDemoPlaying ? "DEMO" : isModeSupported ? stateLabel : "N/A";
  const sessionStatus = !isModeSupported
    ? "Mode offline"
    : !isModelReady
    ? "Loading pose intelligence"
    : isStarted
    ? (mode === "plank" && plankHoldMs > 0 ? "Hold " + formatHoldTime(plankHoldMs) : "Live analysis")
    : isVideoReady
    ? "Camera ready"
    : "Coach idle";
  const fallbackFeedback = isModeSupported
    ? isStarted
      ? currentConfig.activeText
      : currentConfig.standbyText
    : "MediaPipe tracking currently supports the selected labs.";
  const bannerFeedback = isDemoPlaying
    ? "Watch the demo, then start the camera when you are ready."
    : coachFeedback || fallbackFeedback;
  const feedback = appError ? `Error: ${appError}` : bannerFeedback;

  const handleStart = async () => {
    if (isDemoPlaying) {
      setIsDemoPlaying(false);
    }
    await start();
  };

  const handleStop = () => {
    stop();
  };

  const handleModeChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextMode = event.target.value as ExerciseMode;
    setMode(nextMode);
    setIsDemoPlaying(false);
    setDemoKey((key) => key + 1);
    stop();
  };

  const handleShowDemo = () => {
    if (isStarted) {
      stop();
    }
    setIsDemoPlaying(true);
    setDemoKey((key) => key + 1);
  };

  const handleCloseDemo = () => {
    setIsDemoPlaying(false);
  };

  const handleNavigate = (page: "home" | "about") => {
    if (page === activePage) return;
    if (page === "about") {
      stop();
      setIsDemoPlaying(false);
    }
    setActivePage(page);
  };

  useEffect(() => {
    if (mode !== "plank" || !isModeSupported) {
      plankTimerRef.current = null;
      setPlankHoldMs(0);
      return;
    }

    if (!isStarted || isDemoPlaying) {
      plankTimerRef.current = null;
      if (!isStarted) {
        setPlankHoldMs(0);
      }
      return;
    }

    let frame = 0;
    const update = (time: number) => {
      if (plankTimerRef.current === null) {
        plankTimerRef.current = time;
      }
      setPlankHoldMs(time - plankTimerRef.current);
      frame = requestAnimationFrame(update);
    };

    frame = requestAnimationFrame(update);
    return () => {
      cancelAnimationFrame(frame);
      plankTimerRef.current = null;
    };
  }, [mode, isModeSupported, isStarted, isDemoPlaying]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const lines = await fetchExerciseLines(mode);
        if (alive) setTypedLines(lines);
      } catch (error) {
        console.error(error);
      }
    })();
    return () => {
      alive = false;
    };
  }, [mode]);

  return (
    <div className="app-background">
      <div className="app-ribbon">owl hacks 2025</div>
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

          <div className="header-nav" role="navigation" aria-label="Primary">
            <div className="nav-buttons">
              <button
                type="button"
                className={`nav-button ${activePage === "home" ? "is-active" : ""}`}
                onClick={() => handleNavigate("home")}
              >
                Home
              </button>
              <button
                type="button"
                className={`nav-button ${activePage === "about" ? "is-active" : ""}`}
                onClick={() => handleNavigate("about")}
              >
                About
              </button>
            </div>
          </div>

          {activePage === "home" && (
            <div className="header-actions">
              <span className="mode-chip">{currentConfig.label}</span>
              <div className="header-status" aria-live="polite" title={sessionStatus}>
                <span className={`status-dot ${isStarted ? "is-live" : ""}`} />
                <span className="status-text">{statusText}</span>
              </div>
            </div>
          )}
        </header>

        {activePage === "home" ? (
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
                  <span className="stat-label">{metricLabel}</span>
                  <span className="stat-value">{metricValue}</span>
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
                  <span className="video-stage__chip">{stageChipLabel}</span>
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

              <div className={`feedback-banner ${feedback ? "has-text" : ""}`}>
                {feedback}
              </div>
            </section>
          </main>
        ) : (
          <main className="about-page">
            <section className="about-card">
              <h1>About AI Form Coach</h1>
              <p>
                AI Form Coach is our Next Frontier Health submission focused on using computer vision and
                generative AI to bring world-class coaching to any home workout. MediaPipe pose tracking and
                TensorFlow.js power the real-time movement analysis, while Gemini-generated cues keep every
                session encouraging and personalized.
              </p>
              <p>
                By blending motion capture with adaptive feedback, the platform highlights asymmetries,
                poor posture, or wasted effort before they lead to injury. Each demo, cue, and rep counter
                is grounded in biomechanics so people can safely progress—from warm-up to max-effort—in a
                measurable, motivating way.
              </p>
              <p>
                This approach supports Next Frontier Health's mission of sustainable physical wellness:
                accessible coaching, data-backed insights, and AI guidance that scales to every athlete,
                patient, or wellness enthusiast looking to stay active.
              </p>
            </section>
          </main>
        )}

        <footer className="app-footer">
          Built for the Next Frontier Health track with React, TensorFlow.js, and MediaPipe
        </footer>
      </div>
    </div>
  );
}
