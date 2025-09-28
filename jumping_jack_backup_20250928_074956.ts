  const analyzeJumpingJack = useCallback(
    (landmarks: LandmarkList) => {
      const leftAnkle = getPoint(landmarks, 27);
      const rightAnkle = getPoint(landmarks, 28);
      const leftWrist = getPoint(landmarks, 15);
      const rightWrist = getPoint(landmarks, 16);
      const leftShoulder = getPoint(landmarks, 11);
      const rightShoulder = getPoint(landmarks, 12);

      if (!leftAnkle || !rightAnkle || !leftWrist || !rightWrist || !leftShoulder || !rightShoulder) {
        return;
      }

      const ankleGap = distance(leftAnkle, rightAnkle);
      const wristGap = distance(leftWrist, rightWrist);
      const avgShoulderY = (leftShoulder[1] + rightShoulder[1]) / 2;
      const avgWristY = (leftWrist[1] + rightWrist[1]) / 2;
      const armsRaisedDelta = avgShoulderY - avgWristY;
      const now = Date.now();

      const armsHighEnough = armsRaisedDelta >= JACK_ARMS_UP_DELTA;
      const armsDownEnough = armsRaisedDelta <= JACK_ARMS_DOWN_DELTA;
      const legsWideEnough = ankleGap >= JACK_WIDE_ANKLE_GAP;
      const legsCenteredEnough = ankleGap <= JACK_CENTER_ANKLE_GAP;
      const armsWideEnough = wristGap >= JACK_WIDE_WRIST_GAP;
      const armsCenteredEnough = wristGap <= JACK_CENTER_WRIST_GAP;

      const isWide = armsHighEnough && legsWideEnough && armsWideEnough;
      const isCenter = armsDownEnough && legsCenteredEnough && armsCenteredEnough;

      if (isWide) {
        if (jackCenterReadyRef.current) {
          jackCenterReadyRef.current = false;
          setRepCount((previous) => {
            const next = previous + 1;
            speak(`Rep ${next}`, { immediate: true });
            return next;
          });
          setFeedbackMessage("Nice rep - stay tall and keep landing softly.", { silent: true });
        return;
        }
        if (jackState !== "wide") {
          setJackState("wide");
        }
      } else if (isCenter) {
        if (jackState !== "center") {
          setJackState("center");
          setFeedbackMessage("Reset stance, then hit full reach on the next rep.", { silent: true });
        }
        jackCenterReadyRef.current = true;
      } else if (jackCenterReadyRef.current) {
        const armsAlmostHigh = armsRaisedDelta >= JACK_ARMS_UP_DELTA * 0.7;
        const legsAlmostWide = ankleGap >= JACK_WIDE_ANKLE_GAP * 0.75;

        if (!armsHighEnough && legsWideEnough && armsAlmostHigh && now - lastJackFeedbackRef.current > JACK_FEEDBACK_COOLDOWN_MS) {
          setFeedbackMessage("Arms not raised high enough - reach overhead.", { immediate: true });
          lastJackFeedbackRef.current = now;
        } else if (!legsWideEnough && armsHighEnough && legsAlmostWide && now - lastJackFeedbackRef.current > JACK_FEEDBACK_COOLDOWN_MS) {
          setFeedbackMessage("Legs too close - step wider.", { immediate: true });
          lastJackFeedbackRef.current = now;
        }
      }

      if (import.meta.env.DEV) {
        jackDebugCounterRef.current = (jackDebugCounterRef.current + 1) % 12;
        if (jackDebugCounterRef.current === 0) {
          console.debug('[JumpingJack]', {
            ankleGap,
            wristGap,
            armsRaisedDelta,
            armsHighEnough,
            armsDownEnough,
            legsWideEnough,
            legsCenteredEnough,
            armsWideEnough,
            armsCenteredEnough,
            jackCenterReady: jackCenterReadyRef.current,
            jackState,
          });
        }
      }
    },
    [jackState, setFeedbackMessage, speak]
  );
