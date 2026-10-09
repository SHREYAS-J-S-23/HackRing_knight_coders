/**
 * Vidara Ambient Interactive Background Engine
 * - Dynamic Morse code streams on 2D canvas
 * - Mouse parallax tilt & smooth damping
 * - Interactive click sonar waves & luminous particle dispersion
 * - Floating glassmorphic telemetry cards
 */

(function () {
  'use strict';

  // -------------------------------------------------------------
  // 1. Morse Code Signal Canvas
  // -------------------------------------------------------------
  const canvas = document.getElementById('morseCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const MORSE_PATTERNS = [
    '• • •   — — —   • • •',              // SOS / BEACON
    '— • • •   • •   — • •   • —',        // B I D A
    '• — •   • —   — • —   — • —',        // R A K K
    '— — •   • • •   — •   • •',          // G S N I
    '01010110   01001001   01000100',      // VID
    'SYS // 48.0kHz   LOSSLESS',
    'INTEL // 1536D   EMBED',
    'GROQ // 120B   0.28s',
    '— • — •   • — • •   • •   • — — •',  // C L I P
    '• • • —   • •   — • •   • —',        // V I D A
  ];

  // Particle streams
  const streams = [];
  const STREAM_COUNT = Math.min(22, Math.max(12, Math.floor(window.innerWidth / 70)));

  for (let i = 0; i < STREAM_COUNT; i++) {
    streams.push({
      x: Math.random() * width,
      y: Math.random() * height,
      speed: 0.25 + Math.random() * 0.45,
      text: MORSE_PATTERNS[Math.floor(Math.random() * MORSE_PATTERNS.length)],
      opacity: 0.08 + Math.random() * 0.14,
      size: 10 + Math.random() * 3,
      isAccent: Math.random() > 0.65,
      charTimer: 0,
      driftX: (Math.random() - 0.5) * 0.3
    });
  }

  // Interactive mouse tracking
  let mouse = { x: width / 2, y: height / 2, targetX: width / 2, targetY: height / 2 };
  window.addEventListener('mousemove', (e) => {
    mouse.targetX = e.clientX;
    mouse.targetY = e.clientY;
  });

  // Click pulse & particle system
  const clickPulses = [];
  const clickSparks = [];

  window.addEventListener('click', (e) => {
    // 1. Sonar pulse ring
    clickPulses.push({
      x: e.clientX,
      y: e.clientY,
      radius: 0,
      maxRadius: 180 + Math.random() * 60,
      opacity: 0.8,
      speed: 4.5
    });

    // 2. Luminous Morse Sparks
    const sparkCount = 10;
    const symbols = ['•', '—', '0', '1', '+', '•'];
    for (let i = 0; i < sparkCount; i++) {
      const angle = (Math.PI * 2 * i) / sparkCount + (Math.random() - 0.5) * 0.5;
      const velocity = 2.5 + Math.random() * 3.5;
      clickSparks.push({
        x: e.clientX,
        y: e.clientY,
        vx: Math.cos(angle) * velocity,
        vy: Math.sin(angle) * velocity,
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        life: 1.0,
        decay: 0.02 + Math.random() * 0.02,
        size: 10 + Math.random() * 4,
        isOrange: Math.random() > 0.3
      });
    }

    // 3. Jolt floating telemetry cards
    const cards = document.querySelectorAll('.telemetry-card');
    cards.forEach((card) => {
      card.classList.add('pulse-active');
      setTimeout(() => card.classList.remove('pulse-active'), 400);
    });
  });

  // Animation Loop
  let lastTime = 0;
  function animate(timestamp) {
    requestAnimationFrame(animate);

    // Smooth mouse interpolation
    mouse.x += (mouse.targetX - mouse.x) * 0.05;
    mouse.y += (mouse.targetY - mouse.y) * 0.05;

    ctx.clearRect(0, 0, width, height);

    // Parallax delta from center
    const dx = (mouse.x - width / 2) / (width / 2);
    const dy = (mouse.y - height / 2) / (height / 2);

    // 1. Draw Morse Streams
    ctx.font = '500 11px "JetBrains Mono", monospace';
    for (let s of streams) {
      s.y -= s.speed;
      s.x += s.driftX + dx * 0.2;

      // Wrap around
      if (s.y < -30) {
        s.y = height + 30;
        s.x = Math.random() * width;
        s.text = MORSE_PATTERNS[Math.floor(Math.random() * MORSE_PATTERNS.length)];
      }

      if (s.isAccent) {
        ctx.fillStyle = `rgba(255, 106, 26, ${s.opacity})`;
      } else {
        ctx.fillStyle = `rgba(236, 235, 230, ${s.opacity})`;
      }

      ctx.fillText(s.text, s.x, s.y);
    }

    // 2. Draw Sonar Click Pulses
    for (let i = clickPulses.length - 1; i >= 0; i--) {
      const p = clickPulses[i];
      p.radius += p.speed;
      p.opacity = Math.max(0, 1 - p.radius / p.maxRadius);

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 106, 26, ${p.opacity * 0.75})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Outer faint aura
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius * 0.7, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 255, 255, ${p.opacity * 0.25})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      if (p.radius >= p.maxRadius) {
        clickPulses.splice(i, 1);
      }
    }

    // 3. Draw Click Morse Sparks
    for (let i = clickSparks.length - 1; i >= 0; i--) {
      const sp = clickSparks[i];
      sp.x += sp.vx;
      sp.y += sp.vy;
      sp.vx *= 0.94;
      sp.vy *= 0.94;
      sp.life -= sp.decay;

      if (sp.life <= 0) {
        clickSparks.splice(i, 1);
        continue;
      }

      ctx.font = `600 ${sp.size}px "JetBrains Mono", monospace`;
      if (sp.isOrange) {
        ctx.fillStyle = `rgba(255, 106, 26, ${sp.life})`;
      } else {
        ctx.fillStyle = `rgba(236, 235, 230, ${sp.life})`;
      }
      ctx.fillText(sp.symbol, sp.x, sp.y);
    }

    // 4. Parallax on Telemetry Layer
    const telemetryLayer = document.getElementById('floatingTelemetryLayer');
    if (telemetryLayer) {
      const cards = telemetryLayer.querySelectorAll('.telemetry-card');
      cards.forEach((card) => {
        const speed = parseFloat(card.dataset.speed || 1.0);
        const cardDx = -dx * 18 * speed;
        const cardDy = -dy * 14 * speed;
        card.style.transform = `translate3d(${cardDx}px, ${cardDy}px, 0)`;
      });
    }
  }

  requestAnimationFrame(animate);
})();
