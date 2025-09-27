import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import "./App.css";
import { ChevronLeft, ChevronRight, Volume2 } from "lucide-react";


type ExerciseMode = "squat" | "plank";
type SquatPhase = "start" | "up" | "down";

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


export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);

  const [mode, setMode] = useState<ExerciseMode>("squat");
  const [isStarted, setIsStarted] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [repCount, setRepCount] = useState(0);
  const [squatState, setSquatState] = useState<SquatPhase>("start");
  const [feedback, setFeedback] = useState("");

  const lastSpeechTime = useRef(Date.now());
  const speak = useCallback((t: string) => {
    const now = Date.now();
    if (now - lastSpeechTime.current > 3000) {
      speechSynthesis.speak(new SpeechSynthesisUtterance(t));
      lastSpeechTime.current = now;
    }
  }, []);

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

  // --- Frame loop
  useEffect(() => {
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

        if (mode === "squat") analyzeSquat(l);
        else analyzePlank(l);
      }

      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [isStarted, isVideoReady, mode, analyzePlank, analyzeSquat]);

  // --- Cleanup
  useEffect(() => {
    return () => {
      const v = videoRef.current;
      if (v?.srcObject) {
        (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-3xl space-y-5">

        {/* Header */}
        <div className="flex items-center justify-between text-slate-300">
          <ChevronLeft className="opacity-60" />
          <div className="text-center flex-1">
            <h1 className="text-4xl font-extrabold text-white tracking-tight">AI Fitness Coach</h1>
            <p className="text-sm text-slate-400 mt-1">
              Real-time biomechanical analysis powered by MediaPipe
            </p>
          </div>
          <ChevronRight className="opacity-60" />
        </div>

        {/* Mode pills */}
        <div className="flex gap-2">
          <button
            onClick={() => setMode("squat")}
            className={`px-4 py-2 rounded-full text-sm font-semibold transition ${
              mode === "squat"
                ? "bg-indigo-500 text-white shadow-md"
                : "bg-slate-700/70 text-slate-200 hover:bg-slate-600/80"
            }`}
          >
            Squat Mode
          </button>
          <button
            onClick={() => setMode("plank")}
            className={`px-4 py-2 rounded-full text-sm font-semibold transition ${
              mode === "plank"
                ? "bg-indigo-500 text-white shadow-md"
                : "bg-slate-700/70 text-slate-200 hover:bg-slate-600/80"
            }`}
          >
            Plank Mode
          </button>
        </div>

        {/* Stat bar */}
        <div className="flex items-center justify-between bg-slate-800/80 rounded-full px-4 py-3 stat-shadow">
          <div className="flex items-center gap-3">
            <span className="bg-amber-400 text-slate-900 font-bold px-4 py-1.5 rounded-full">
              Reps: {repCount}
            </span>
            <span className="text-slate-200 font-semibold">
              State: {mode === "squat" ? (squatState.toUpperCase()) : "HOLD"}
            </span>
          </div>
          <button
            className="w-9 h-9 rounded-full bg-slate-700/80 grid place-items-center hover:bg-slate-600"
            title="Voice feedback"
          >
            <Volume2 size={18} className="text-slate-200" />
          </button>
        </div>

        {/* Video frame */}
        <div className="relative video-shell rounded-[22px] p-4">
          <div className="video-inner rounded-[16px] overflow-hidden">
            <div className="video-container">
              <video ref={videoRef} playsInline muted />
              <canvas ref={canvasRef} />
            </div>
          </div>

          {/* Start/Stop button */}
          {!isStarted ? (
            <button
              onClick={handleStart}
              className="absolute bottom-4 right-4 px-5 py-2 rounded-full bg-emerald-500 text-white font-semibold shadow hover:bg-emerald-600"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="absolute bottom-4 right-4 px-5 py-2 rounded-full bg-rose-500 text-white font-semibold shadow hover:bg-rose-600"
            >
              Stop Camera
            </button>
          )}
        </div>

        {/* Feedback pill */}
        <div className="rounded-full px-6 py-3 bg-emerald-200 text-emerald-900 font-semibold w-fit mx-auto shadow-sm">
          {feedback ? feedback : (isStarted ? "Analyzing…" : "Click Start to begin.")}
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-slate-400">
          Built for the Next Frontier Health track using React and Google MediaPipe
        </p>
      </div>
    </div>
  );

}
