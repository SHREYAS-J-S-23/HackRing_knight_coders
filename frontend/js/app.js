/**
 * VIDARA — AI Video Intelligence Platform Controller
 * "Understand Every Moment."
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    selectedFile: null,
    videoId: null,
    duration: 0,
    transcript: null,
    coreThesis: '',
    topics: [],
    allDiscoveredTopics: [],
    isQueryMode: false,
    selectedTopicIds: new Set(),
    clips: [],
    selectedClipIds: new Set(),
    isVoiceRecording: false,
    speechRecognition: null,
    audiences: [],
    selectedAudienceId: null,
    currentEdl: null,
    validationReport: null,
    overrides: {},
    mergedVideoUrl: '',
    mergedVttUrl: '',
    apiBase: window.location.origin,
    token: localStorage.getItem('vidara_token') || null,
    user: JSON.parse(localStorage.getItem('vidara_user') || 'null'),
    videoTitle: '',
    activeClip: null
  };

  // Auth & User Profile Elements
  const authBtn = document.getElementById('authBtn');
  const userProfilePill = document.getElementById('userProfilePill');
  const userAvatarMini = document.getElementById('userAvatarMini');
  const userNameLabel = document.getElementById('userNameLabel');
  const logoutBtn = document.getElementById('logoutBtn');
  const authModal = document.getElementById('authModal');
  const closeAuthModalBtn = document.getElementById('closeAuthModalBtn');
  const tabAuthPhone = document.getElementById('tabAuthPhone');
  const tabAuthGoogle = document.getElementById('tabAuthGoogle');
  const panelAuthPhone = document.getElementById('panelAuthPhone');
  const panelAuthGoogle = document.getElementById('panelAuthGoogle');
  const authNameInput = document.getElementById('authNameInput');
  const authPhoneInput = document.getElementById('authPhoneInput');
  const sendOtpBtn = document.getElementById('sendOtpBtn');
  const otpSection = document.getElementById('otpSection');
  const otpDevBanner = document.getElementById('otpDevBanner');
  const devOtpBadge = document.getElementById('devOtpBadge');
  const authOtpInput = document.getElementById('authOtpInput');
  const verifyOtpBtn = document.getElementById('verifyOtpBtn');
  const directGoogleLoginBtn = document.getElementById('directGoogleLoginBtn');

  // Library & Saved Clips Elements
  const navLibraryBtn = document.getElementById('navLibraryBtn');
  const navLibraryCountBadge = document.getElementById('navLibraryCountBadge');
  const libraryModal = document.getElementById('libraryModal');
  const closeLibraryModalBtn = document.getElementById('closeLibraryModalBtn');
  const tabLibClips = document.getElementById('tabLibClips');
  const tabLibVideos = document.getElementById('tabLibVideos');
  const panelLibClips = document.getElementById('panelLibClips');
  const panelLibVideos = document.getElementById('panelLibVideos');
  const libClipsCountBadge = document.getElementById('libClipsCountBadge');
  const libVideosCountBadge = document.getElementById('libVideosCountBadge');
  const savedClipsGrid = document.getElementById('savedClipsGrid');
  const emptyLibraryClipsMsg = document.getElementById('emptyLibraryClipsMsg');
  const savedVideosList = document.getElementById('savedVideosList');
  const emptyLibraryVideosMsg = document.getElementById('emptyLibraryVideosMsg');
  const saveCurrentStudioClipBtn = document.getElementById('saveCurrentStudioClipBtn');

  // DOM Elements
  const tabUploadFile = document.getElementById('tabUploadFile');
  const tabUploadLink = document.getElementById('tabUploadLink');
  const panelUploadFile = document.getElementById('panelUploadFile');
  const panelUploadLink = document.getElementById('panelUploadLink');
  const videoUrlInput = document.getElementById('videoUrlInput');
  const fetchVideoUrlBtn = document.getElementById('fetchVideoUrlBtn');

  const dropzone = document.getElementById('videoDropzone');
  const fileInput = document.getElementById('videoFileInput');
  const browseFileBtn = document.getElementById('browseFileBtn');
  const useSampleDemoBtn = document.getElementById('useSampleDemoBtn');
  const videoPreviewBar = document.getElementById('videoPreviewBar');
  const videoFileName = document.getElementById('videoFileName');
  const videoFileStats = document.getElementById('videoFileStats');
  const uploadQuestionInput = document.getElementById('queryTextInput');
  const startPipelineBtn = document.getElementById('startPipelineBtn');

  // Mode Elements
  const queryTextInput = document.getElementById('queryTextInput');
  const voiceQueryBtn = document.getElementById('voiceQueryBtn');
  const voiceStatusPill = document.getElementById('voiceStatusPill');
  const submitQueryBtn = document.getElementById('submitQueryBtn');
  const discoverTopicsBtn = document.getElementById('discoverTopicsBtn');
  const autoExtractHighlightsBtn = document.getElementById('autoExtractHighlightsBtn');
  const topicsDashboardTitle = document.getElementById('topicsDashboardTitle');
  const showAllScenesBtn = document.getElementById('showAllScenesBtn');
  const allScenesCount = document.getElementById('allScenesCount');
  const samplePromptTags = document.querySelectorAll('.sample-prompt-tag');

  // Processing & Stepper Elements
  const processingCard = document.getElementById('processingCard');
  const processingTitle = document.getElementById('processingTitle');
  const processingDetail = document.getElementById('processingDetail');
  const progressBar = document.getElementById('progressBar');
  const stagePills = {
    upload: document.getElementById('stgUpload'),
    audio: document.getElementById('stgAudio'),
    transcribe: document.getElementById('stgTranscribe'),
    index: document.getElementById('stgIndex'),
    topics: document.getElementById('stgTopics'),
    rank: document.getElementById('stgRank'),
    boundaries: document.getElementById('stgBoundaries'),
    ready: document.getElementById('stgReady')
  };

  // Results & Dashboard Elements
  const queryResultsCard = document.getElementById('queryResultsCard');
  const queryConceptsPills = document.getElementById('queryConceptsPills');
  const queryReasoningText = document.getElementById('queryReasoningText');

  const topicsDashboardCard = document.getElementById('topicsDashboardCard');
  const rankedTopicsGrid = document.getElementById('rankedTopicsGrid');
  const selectAllTopicsCheckbox = document.getElementById('selectAllTopicsCheckbox');
  const selectedCountBadge = document.getElementById('selectedCountBadge');
  const generateSelectedClipsBtn = document.getElementById('generateSelectedClipsBtn');
  const mergeSelectedClipsBtn = document.getElementById('mergeSelectedClipsBtn');

  // Advanced / Output Studio Elements
  const coreThesisText = document.getElementById('coreThesisText');
  const audienceGrid = document.getElementById('audienceGrid');

  const outputSection = document.getElementById('outputSection');
  const finalVideoPlayer = document.getElementById('finalVideoPlayer');
  const videoSubtitlesTrack = document.getElementById('videoSubtitlesTrack');
  const downloadVideoBtn = document.getElementById('downloadVideoBtn');
  const downloadSubtitlesBtn = document.getElementById('downloadSubtitlesBtn');
  const downloadAuditLogBtn = document.getElementById('downloadAuditLogBtn');
  const quickMergeSelectedBtn = document.getElementById('quickMergeSelectedBtn');

  const metricOriginalDuration = document.getElementById('metricOriginalDuration');
  const metricRepurposedDuration = document.getElementById('metricRepurposedDuration');
  const metricCompression = document.getElementById('metricCompression');
  const metricCompressionDetail = document.getElementById('metricCompressionDetail');
  const faithfulnessScore = document.getElementById('faithfulnessScore');

  const tabMergedVideo = document.getElementById('tabMergedVideo');
  const tabIndividualClips = document.getElementById('tabIndividualClips');
  const clipsCountBadge = document.getElementById('clipsCountBadge');
  const nowPlayingLabel = document.getElementById('nowPlayingLabel');
  const downloadVideoBtnText = document.getElementById('downloadVideoBtnText');
  const studioPlaylistContainer = document.getElementById('studioPlaylistContainer');
  const studioClipsList = document.getElementById('studioClipsList');
  const individualClipsPanel = document.getElementById('individualClipsPanel');
  const clipsGrid = document.getElementById('clipsGrid');
  const mergeClipsActionBtn = document.getElementById('mergeClipsActionBtn');

  // Reset and purge any browser autofill on query search bar
  function purgeAutofill() {
    if (queryTextInput) {
      const val = (queryTextInput.value || '').trim();
      if (val.includes('@') || val.toLowerCase().includes('karanth') || val.toLowerCase().includes('vishnu')) {
        queryTextInput.value = '';
      }
    }
  }

  if (queryTextInput) {
    queryTextInput.value = '';
    queryTextInput.addEventListener('focus', purgeAutofill);
    queryTextInput.addEventListener('input', purgeAutofill);
    setTimeout(purgeAutofill, 50);
    setTimeout(purgeAutofill, 300);
    setTimeout(purgeAutofill, 1000);
  }

  async function parseErrorResponse(res, defaultMsg) {
    try {
      const text = await res.text();
      try {
        const data = JSON.parse(text);
        return data.detail || data.message || defaultMsg;
      } catch {
        return (text && text.length < 300) ? text : defaultMsg;
      }
    } catch {
      return defaultMsg;
    }
  }

  // Initialize Speech Recognition, Audiences, Auth & Library
  initSpeechRecognition();
  fetchAudiences();
  initAuthUI();
  initLibraryUI();
  checkAuthState();

  // -------------------------------------------------------------
  // 1. VIDEO INGESTION: FILE UPLOAD & LINK IMPORT
  // -------------------------------------------------------------
  // Mode Tabs Switching
  if (tabUploadFile && tabUploadLink) {
    tabUploadFile.addEventListener('click', () => {
      tabUploadFile.classList.add('active');
      tabUploadLink.classList.remove('active');
      if (panelUploadFile) panelUploadFile.classList.remove('hidden');
      if (panelUploadLink) panelUploadLink.classList.add('hidden');
    });

    tabUploadLink.addEventListener('click', () => {
      tabUploadLink.classList.add('active');
      tabUploadFile.classList.remove('active');
      if (panelUploadLink) panelUploadLink.classList.remove('hidden');
      if (panelUploadFile) panelUploadFile.classList.add('hidden');
      if (videoUrlInput) videoUrlInput.focus();
    });
  }

  // Link Ingestion Actions
  if (fetchVideoUrlBtn && videoUrlInput) {
    fetchVideoUrlBtn.addEventListener('click', async () => {
      await importAndIndexVideoUrl(videoUrlInput.value.trim());
    });

    videoUrlInput.addEventListener('keydown', async (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        await importAndIndexVideoUrl(videoUrlInput.value.trim());
      }
    });
  }

  browseFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('#useSampleDemoBtn')) return;
    fileInput.click();
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    state.selectedFile = file;
    videoFileName.textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    videoFileStats.textContent = `${sizeMb} MB • Ready for Vidara intelligence`;
    videoPreviewBar.classList.remove('hidden');
    dropzone.style.display = 'none';
  }

  // Sample Founder Video Demo Button
  useSampleDemoBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    setProcessing(true, "Loading Sample Video...", "Fetching raw video recording...", 15, 'upload');
    try {
      const response = await fetch(`${state.apiBase}/static/uploads/sample_raw.mp4`);
      if (!response.ok) throw new Error("Sample file not available.");
      const blob = await response.blob();
      const sampleFile = new File([blob], 'founder_walkthrough_raw.mp4', { type: 'video/mp4' });
      handleFileSelected(sampleFile);
      setProcessing(false);
      startPipelineBtn.click();
    } catch (e) {
      console.warn("Falling back to local video file selection:", e);
      setProcessing(false);
      fileInput.click();
    }
  });

  // Start Pipeline / Indexing Button
  startPipelineBtn.addEventListener('click', async () => {
    if (!state.selectedFile) return;
    await uploadAndIndexVideo();
  });

  async function runAnalysisPipeline(videoId, duration) {
    setProcessing(true, "Transcribing with Whisper Large-v3...", "Generating word-level millisecond timestamps on Groq LPUs...", 20, 'transcribe');
    updateStepIndicator(2);
    setStagePill('transcribe');

    // STEP 1: Fire the analyze request — returns immediately (202 queued or 200 already-indexed)
    const kickoffRes = await fetch(`${state.apiBase}/api/videos/${videoId}/analyze`, {
      method: 'POST'
    });
    if (!kickoffRes.ok) {
      const errMsg = await parseErrorResponse(kickoffRes, "Video analysis kickoff failed");
      throw new Error(errMsg);
    }
    const kickoffData = await kickoffRes.json();

    // If the video was already indexed in a previous run, skip polling
    let analyzeData = null;
    if (kickoffData.status === 'success' && kickoffData.already_indexed) {
      analyzeData = kickoffData;
    } else {
      // STEP 2: Poll /status until the background job completes
      analyzeData = await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          try {
            const sRes = await fetch(`${state.apiBase}/api/videos/${videoId}/status`);
            if (!sRes.ok) return;
            const sData = await sRes.json();

            // Live progress display
            const stage = sData.stage || 'Processing';
            const msg = sData.message || 'Processing video intelligence...';
            const progress = Math.max(20, sData.progress || 30);
            setProcessing(true, `${stage}...`, msg, progress, 'transcribe');

            if (sData.status === 'completed') {
              clearInterval(poller);
              // STEP 3: Fetch the full result once done
              try {
                const resultRes = await fetch(`${state.apiBase}/api/videos/${videoId}/analyze-result`);
                if (!resultRes.ok) {
                  const errText = await resultRes.text();
                  reject(new Error(`Result fetch failed: ${errText}`));
                  return;
                }
                const resultData = await resultRes.json();
                resolve(resultData);
              } catch (e) {
                reject(e);
              }
            } else if (sData.status === 'failed') {
              clearInterval(poller);
              reject(new Error(sData.message || 'Analysis pipeline failed'));
            }
          } catch (e) {
            // Network hiccup — keep polling
          }
        }, 1500);
      });
    }

    // analyzeData now holds the full result — same shape as before
    state.coreThesis = analyzeData.core_thesis;
    if (coreThesisText) coreThesisText.textContent = state.coreThesis;

    // Update Video Intelligence Overview Card
    const videoOverviewCard = document.getElementById('videoOverviewCard');
    const summaryDuration = document.getElementById('summaryDuration');
    const summarySegments = document.getElementById('summarySegments');
    const summaryTopics = document.getElementById('summaryTopics');
    const overviewCoreThesis = document.getElementById('overviewCoreThesis');

    if (summaryDuration) summaryDuration.textContent = formatTime(duration);
    if (summarySegments) summarySegments.textContent = analyzeData.segment_count || '0';
    if (overviewCoreThesis) overviewCoreThesis.textContent = state.coreThesis || 'Complete video indexed.';

    // Save all discovered topics from video analysis
    state.allDiscoveredTopics = analyzeData.topics || [];
    state.topics = analyzeData.topics || [];
    if (summaryTopics) summaryTopics.textContent = analyzeData.topic_count || state.topics.length || '0';
    if (videoOverviewCard) videoOverviewCard.classList.remove('hidden');

    setStagePill('ready');
    setProcessing(false);
    updateStepIndicator(3);

    // Render all separate scenes/topics from the video with clip cuts and option to merge!
    renderAllScenes(state.allDiscoveredTopics);

    setTimeout(() => {
      if (topicsDashboardCard) topicsDashboardCard.scrollIntoView({ behavior: 'smooth' });
    }, 300);
  }

  async function uploadAndIndexVideo() {
    setProcessing(true, "Ingesting Video...", "Uploading large file and extracting audio track with FFmpeg...", 20, 'upload');
    updateStepIndicator(1);
    setStagePill('upload');

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    state.videoTitle = state.selectedFile ? state.selectedFile.name : 'Uploaded Video';

    try {
      setStagePill('audio');
      const uploadHeaders = {};
      if (state.token) {
        uploadHeaders['Authorization'] = `Bearer ${state.token}`;
      }
      const uploadRes = await fetch(`${state.apiBase}/api/videos/upload`, {
        method: 'POST',
        headers: uploadHeaders,
        body: formData
      });

      if (!uploadRes.ok) {
        const errMsg = await parseErrorResponse(uploadRes, "Video upload failed");
        throw new Error(errMsg);
      }

      const uploadData = await uploadRes.json();
      state.videoId = uploadData.video_id;
      state.duration = uploadData.duration_seconds;

      await runAnalysisPipeline(state.videoId, state.duration);

    } catch (err) {
      alert(`Vidara Pipeline Error: ${err.message}`);
      setProcessing(false);
    }
  }

  async function importAndIndexVideoUrl(url) {
    if (!url || !url.trim()) {
      alert("Please enter a valid video link.");
      return;
    }
    const cleanUrl = url.trim();
    if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
      alert("Please provide a valid web link starting with http:// or https://");
      return;
    }

    setProcessing(true, "Importing Video from Link...", "Downloading stream with yt-dlp & extracting audio with FFmpeg...", 20, 'upload');
    updateStepIndicator(1);
    setStagePill('upload');

    try {
      setStagePill('audio');
      const ingestHeaders = { 'Content-Type': 'application/json' };
      if (state.token) {
        ingestHeaders['Authorization'] = `Bearer ${state.token}`;
      }
      const ingestRes = await fetch(`${state.apiBase}/api/videos/ingest-url`, {
        method: 'POST',
        headers: ingestHeaders,
        body: JSON.stringify({ url: cleanUrl })
      });

      if (!ingestRes.ok) {
        const errMsg = await parseErrorResponse(ingestRes, "Video link ingestion failed");
        throw new Error(errMsg);
      }

      const ingestData = await ingestRes.json();
      state.videoId = ingestData.video_id;
      state.duration = ingestData.duration_seconds;
      state.videoTitle = ingestData.filename || 'Imported Video';

      if (videoFileName) videoFileName.textContent = ingestData.filename;
      if (videoFileStats) videoFileStats.textContent = `${formatTime(state.duration)} • Imported from link`;
      if (videoPreviewBar) videoPreviewBar.classList.remove('hidden');

      // Seamlessly execute Whisper Transcription & Semantic Graph Indexing
      await runAnalysisPipeline(state.videoId, state.duration);

    } catch (err) {
      alert(`Link Import Error: ${err.message}`);
      setProcessing(false);
    }
  }

  // -------------------------------------------------------------
  // 2. MODE A: ASK VIDARA (TEXT OR VOICE)
  // -------------------------------------------------------------
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        state.isVoiceRecording = true;
        voiceQueryBtn.classList.add('recording');
        voiceStatusPill.classList.remove('hidden');
      };

      recognition.onresult = (event) => {
        const transcriptText = event.results[0][0].transcript;
        queryTextInput.value = transcriptText;
        voiceStatusPill.classList.add('hidden');
        state.isVoiceRecording = false;
        voiceQueryBtn.classList.remove('recording');
        // Auto-trigger search
        submitQueryBtn.click();
      };

      recognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        voiceStatusPill.classList.add('hidden');
        state.isVoiceRecording = false;
        voiceQueryBtn.classList.remove('recording');
      };

      recognition.onend = () => {
        voiceStatusPill.classList.add('hidden');
        state.isVoiceRecording = false;
        voiceQueryBtn.classList.remove('recording');
      };

      state.speechRecognition = recognition;
    } else {
      voiceQueryBtn.title = "Voice recognition not supported in this browser. Please type your query.";
    }
  }

  voiceQueryBtn.addEventListener('click', () => {
    if (!state.speechRecognition) {
      alert("Speech recognition is not supported in this browser. Please type your query in the search bar.");
      return;
    }
    if (state.isVoiceRecording) {
      state.speechRecognition.stop();
    } else {
      state.speechRecognition.start();
    }
  });

  samplePromptTags.forEach(tag => {
    tag.addEventListener('click', () => {
      queryTextInput.value = tag.getAttribute('data-query');
      submitQueryBtn.click();
    });
  });



  queryTextInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitQueryBtn.click();
    }
  });

  async function executeVideoQuery(query) {
    if (!state.videoId) {
      alert("Please select and index a video first.");
      document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
      return;
    }

    setProcessing(true, "Searching Video Moments...", `Finding exact scene for: "${query}"...`, 80, 'rank');
    updateStepIndicator(3);

    try {
      const res = await fetch(`${state.apiBase}/api/videos/${state.videoId}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, voice_input: false })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      renderQueryResults(data);
      setProcessing(false);
      setStagePill('ready');
      updateStepIndicator(4);

      topicsDashboardCard.classList.remove('hidden');
      topicsDashboardCard.scrollIntoView({ behavior: 'smooth' });

    } catch (e) {
      alert(`Query Error: ${e.message}`);
      setProcessing(false);
    }
  }

  submitQueryBtn.addEventListener('click', async () => {
    let query = queryTextInput.value.trim();
    if (!query) {
      // If blank query, switch back to all scenes
      renderAllScenes(state.allDiscoveredTopics);
      queryResultsCard.classList.add('hidden');
      return;
    }
    await executeVideoQuery(query);
  });

  function renderQueryResults(data) {
    queryResultsCard.classList.remove('hidden');
    queryConceptsPills.innerHTML = '';

    const isFound = data.found !== false && (data.matched_topics && data.matched_topics.length > 0);
    const queryResultBadge = document.getElementById('queryResultBadge');

    if (isFound) {
      if (queryResultBadge) {
        queryResultBadge.className = 'badge badge-accent';
        queryResultBadge.textContent = 'Moment Located';
      }

      (data.understood_concepts || []).forEach(concept => {
        const badge = document.createElement('span');
        badge.className = 'concept-badge';
        badge.textContent = concept;
        queryConceptsPills.appendChild(badge);
      });

      queryReasoningText.innerHTML = `<span style="color:var(--color-green);font-weight:600;"><i class="fa-solid fa-circle-check"></i> Grounded Match:</span> ${escapeHtml(data.reasoning || "Vidara retrieved matched sections.")}`;

      // Targeted moment mode: display ONLY the matching scene(s)
      state.isQueryMode = true;
      state.topics = data.matched_topics || [];

      if (topicsDashboardTitle) {
        topicsDashboardTitle.innerHTML = `<i class="fa-solid fa-bullseye icon-accent"></i> TARGETED SCENE: "${escapeHtml(data.query)}"`;
      }
    } else {
      // Content does not exist in the video!
      if (queryResultBadge) {
        queryResultBadge.className = 'badge badge-warning';
        queryResultBadge.textContent = 'Not in Video';
      }

      const emptyConceptBadge = document.createElement('span');
      emptyConceptBadge.className = 'concept-badge';
      emptyConceptBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      emptyConceptBadge.style.color = '#fbbf24';
      emptyConceptBadge.innerHTML = '<i class="fa-solid fa-ban"></i> No Matching Moments';
      queryConceptsPills.appendChild(emptyConceptBadge);

      queryReasoningText.innerHTML = `
        <div class="no-content-alert-box">
          <i class="fa-solid fa-triangle-exclamation"></i>
          <div>
            <strong>No Such Content in This Video</strong>
            <p>${escapeHtml(data.reasoning || `Vidara verified the transcript: there is no content or discussion regarding "${data.query}" in this video.`)}</p>
          </div>
        </div>
      `;

      state.isQueryMode = true;
      state.topics = [];

      if (topicsDashboardTitle) {
        topicsDashboardTitle.innerHTML = `<i class="fa-solid fa-circle-exclamation icon-accent" style="color:#fbbf24;"></i> NO MATCHING SCENES FOR: "${escapeHtml(data.query)}"`;
      }
    }

    if (showAllScenesBtn) {
      showAllScenesBtn.classList.remove('hidden');
      if (allScenesCount) allScenesCount.textContent = (state.allDiscoveredTopics || []).length;
    }

    renderRankedTopics(state.topics);
  }

  function renderAllScenes(topics) {
    state.isQueryMode = false;
    state.topics = (topics && topics.length > 0) ? topics : (state.allDiscoveredTopics || []);
    if (topicsDashboardTitle) {
      topicsDashboardTitle.innerHTML = '<i class="fa-solid fa-list-check icon-accent"></i> VIDARA DISCOVERED TOPICS';
    }
    if (showAllScenesBtn) {
      showAllScenesBtn.classList.add('hidden');
    }
    renderRankedTopics(state.topics);
    topicsDashboardCard.classList.remove('hidden');
  }

  if (showAllScenesBtn) {
    showAllScenesBtn.addEventListener('click', () => {
      queryTextInput.value = '';
      if (uploadQuestionInput) uploadQuestionInput.value = '';
      renderAllScenes(state.allDiscoveredTopics);
      queryResultsCard.classList.add('hidden');
    });
  }

  async function triggerAutonomousDiscovery() {
    if (!state.videoId) {
      alert("Please upload or import a video first.");
      document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
      return;
    }

    setProcessing(true, "Discovering High-Value Moments...", "Analyzing conversation turns, Q&A dynamics, speaker boundaries, and editorial value...", 70, 'topics');
    updateStepIndicator(3);
    setStagePill('topics');

    try {
      const res = await fetch(`${state.apiBase}/api/videos/${state.videoId}/discover-topics`, {
        method: 'POST'
      });

      if (!res.ok) {
        const errMsg = await parseErrorResponse(res, "Topic discovery failed");
        throw new Error(errMsg);
      }

      const data = await res.json();
      state.topics = data.topics || [];
      state.allDiscoveredTopics = state.topics;

      setStagePill('rank');
      setProcessing(true, "Validating Conversational Boundaries...", "Applying 10-point editorial audit & conciseness optimization...", 90, 'boundaries');

      setTimeout(() => {
        setProcessing(false);
        setStagePill('ready');
        updateStepIndicator(4);
        queryResultsCard.classList.add('hidden');
        renderAllScenes(state.topics);
        topicsDashboardCard.classList.remove('hidden');
        topicsDashboardCard.scrollIntoView({ behavior: 'smooth' });
      }, 500);

    } catch (e) {
      alert(`Discovery Error: ${e.message}`);
      setProcessing(false);
    }
  }

  if (autoExtractHighlightsBtn) {
    autoExtractHighlightsBtn.addEventListener('click', async () => {
      queryTextInput.value = '';
      if (uploadQuestionInput) uploadQuestionInput.value = '';
      if (state.allDiscoveredTopics && state.allDiscoveredTopics.length > 0) {
        renderAllScenes(state.allDiscoveredTopics);
        queryResultsCard.classList.add('hidden');
        topicsDashboardCard.classList.remove('hidden');
        topicsDashboardCard.scrollIntoView({ behavior: 'smooth' });
      } else {
        await triggerAutonomousDiscovery();
      }
    });
  }

  // -------------------------------------------------------------
  // 3. MODE B: DISCOVER IMPORTANT TOPICS (AUTONOMOUS)
  // -------------------------------------------------------------
  if (discoverTopicsBtn) {
    discoverTopicsBtn.addEventListener('click', async () => {
      await triggerAutonomousDiscovery();
    });
  }

  // -------------------------------------------------------------
  // 4. TOPIC DASHBOARD & SELECTION MANAGEMENT
  // -------------------------------------------------------------
  function renderRankedTopics(topics) {
    rankedTopicsGrid.innerHTML = '';
    state.selectedTopicIds.clear();

    if (!topics || topics.length === 0) {
      // Immediately stop any video playback and hide studio so no random video is playing!
      if (finalVideoPlayer) {
        try { finalVideoPlayer.pause(); } catch (_) {}
      }
      if (outputSection) {
        outputSection.classList.add('hidden');
      }
      if (generateSelectedClipsBtn) generateSelectedClipsBtn.classList.add('hidden');
      if (mergeSelectedClipsBtn) mergeSelectedClipsBtn.classList.add('hidden');
      if (selectionPillWrapper) selectionPillWrapper.classList.add('hidden');

      if (state.isQueryMode) {
        const queryVal = (queryTextInput && queryTextInput.value.trim()) ? queryTextInput.value.trim() : 'your request';
        rankedTopicsGrid.innerHTML = `
          <div class="no-content-card">
            <div class="no-content-icon"><i class="fa-solid fa-file-circle-xmark"></i></div>
            <h3 class="no-content-title">There is No Such Content in This Video</h3>
            <p class="no-content-desc">
              Vidara analyzed the video transcript and intent graph, but detected no moments discussing <strong>"${escapeHtml(queryVal)}"</strong>.
              The speaker does not cover this topic in the uploaded recording.
            </p>
            <div class="no-content-actions">
              <button type="button" id="returnAllTopicsBtn" class="btn-primary">
                <i class="fa-solid fa-list-check"></i> View Discovered Video Topics (${(state.allDiscoveredTopics || []).length})
              </button>
            </div>
          </div>
        `;
        const retBtn = document.getElementById('returnAllTopicsBtn');
        if (retBtn) {
          retBtn.addEventListener('click', () => {
            if (queryTextInput) queryTextInput.value = '';
            renderAllScenes(state.allDiscoveredTopics);
            queryResultsCard.classList.add('hidden');
          });
        }
      } else {
        rankedTopicsGrid.innerHTML = '<p class="text-secondary">No topics found. Try another query.</p>';
      }
      updateSelectionCounter();
      return;
    }

    // Restore generation actions when valid topics exist
    if (generateSelectedClipsBtn) generateSelectedClipsBtn.classList.remove('hidden');
    if (mergeSelectedClipsBtn) mergeSelectedClipsBtn.classList.remove('hidden');
    if (selectionPillWrapper) selectionPillWrapper.classList.remove('hidden');

    topics.forEach((t, idx) => {
      state.selectedTopicIds.add(t.id);

      const card = document.createElement('div');
      card.className = 'topic-card selected';
      card.id = `card_${t.id}`;

      const rankStr = String(idx + 1).padStart(2, '0');
      const startMinSec = formatTime(t.start_time);
      const endMinSec = formatTime(t.end_time);

      const subtopicsHtml = (t.subtopics || []).slice(0, 4).map(sub => `
        <span class="subtopic-item"><i class="fa-solid fa-check text-green"></i> ${escapeHtml(sub)}</span>
      `).join('');

      const whyHtml = (t.why_selected || []).slice(0, 4).map(w => `
        <div class="why-item"><i class="fa-solid fa-check"></i> <span>${escapeHtml(w)}</span></div>
      `).join('');

      card.innerHTML = `
        <div>
          <div class="topic-header-row">
            <div>
              <span class="topic-rank-num">${rankStr}</span>
              <h3 class="topic-name">${escapeHtml(t.name)}</h3>
            </div>
            <span class="topic-time-badge"><i class="fa-regular fa-clock"></i> ${startMinSec} — ${endMinSec}</span>
          </div>

          <div class="topic-scores-strip" style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <span class="score-pill score-importance">Importance ${(t.importance_score * 100).toFixed(0)}%</span>
            <span class="score-pill score-confidence">Confidence ${(t.confidence * 100).toFixed(0)}%</span>
            ${(t.exchange_type && t.exchange_type !== 'GENERAL_TOPIC') ? `
              <span class="badge badge-podcast">
                <i class="fa-solid fa-podcast"></i> ${escapeHtml(t.exchange_type.replace(/_/g, ' '))}
              </span>
            ` : ''}
          </div>

          ${(t.speakers_involved && t.speakers_involved.length > 0) ? `
          <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; align-items: center;">
            ${t.speakers_involved.map(spk => {
              const role = (t.speaker_roles && t.speaker_roles[spk]) ? ` • ${t.speaker_roles[spk]}` : '';
              return `<span class="badge badge-speaker"><i class="fa-solid fa-user"></i> ${escapeHtml(spk)}${role}</span>`;
            }).join('')}
            ${t.question_included ? `<span class="badge badge-success" style="font-size:0.7rem; padding:1px 6px;"><i class="fa-solid fa-circle-question"></i> Question Preserved</span>` : ''}
          </div>
          ` : ''}

          ${t.editorial_justification ? `
          <div style="margin-top: 8px; font-size: 0.78rem; color: #c7d2fe; background: rgba(99, 102, 241, 0.08); padding: 6px 10px; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-primary); line-height: 1.4;">
            <i class="fa-solid fa-feather-pointed"></i> <strong>Editorial Insight:</strong> ${escapeHtml(t.editorial_justification)}
          </div>
          ` : ''}

          <div class="subtopics-box">
            <div class="subtopics-label">Subtopics & Concepts</div>
            <div class="subtopics-badges">${subtopicsHtml || '<span class="subtopic-item">Core Concept</span>'}</div>
          </div>

          ${t.key_information ? `
          <div class="subtopics-box" style="margin-top: 6px;">
            <div class="subtopics-label"><i class="fa-solid fa-bolt icon-accent"></i> Key Insight</div>
            <p style="font-size:0.82rem; color:var(--text-secondary); margin: 3px 0 0 0; line-height: 1.4;">${escapeHtml(t.key_information)}</p>
          </div>
          ` : ''}

          <div class="why-selected-box">
            <div class="why-selected-label"><i class="fa-solid fa-sparkles"></i> Why Vidara Selected This</div>
            <div class="why-selected-list">${whyHtml}</div>
          </div>
        </div>

        <div class="topic-card-actions">
          <label class="topic-select-label">
            <input type="checkbox" class="topic-checkbox" data-topic-id="${t.id}" checked />
            <span>Select</span>
          </label>
          <div style="display: flex; gap: 8px;">
            <button class="btn-ghost btn-sm preview-topic-btn" data-start="${t.start_time}" data-end="${t.end_time}" data-title="${escapeHtml(t.name)}">
              <i class="fa-solid fa-play"></i> Preview
            </button>
            <button class="btn-ghost btn-sm save-topic-btn" data-topic-id="${t.id}" data-title="${escapeHtml(t.name)}" data-start="${t.start_time}" data-end="${t.end_time}" title="Save clip permanently to My Library">
              <i class="fa-regular fa-bookmark"></i> Save
            </button>
            <button class="btn-primary btn-sm generate-single-clip-btn" data-topic-id="${t.id}">
              <i class="fa-solid fa-scissors"></i> Clip
            </button>
          </div>
        </div>
      `;

      rankedTopicsGrid.appendChild(card);
    });

    updateSelectionCounter();
    bindTopicCardEvents();
  }

  function bindTopicCardEvents() {
    // Topic card checkboxes
    document.querySelectorAll('.topic-checkbox').forEach(cb => {
      cb.addEventListener('change', (e) => {
        const tid = e.target.getAttribute('data-topic-id');
        const card = document.getElementById(`card_${tid}`);
        if (e.target.checked) {
          state.selectedTopicIds.add(tid);
          if (card) card.classList.add('selected');
        } else {
          state.selectedTopicIds.delete(tid);
          if (card) card.classList.remove('selected');
        }
        updateSelectionCounter();
      });
    });

    // Preview button
    document.querySelectorAll('.preview-topic-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const start = parseFloat(btn.getAttribute('data-start') || '0');
        const end = parseFloat(btn.getAttribute('data-end') || '0');
        const title = btn.getAttribute('data-title') || '';
        previewAtTimestamp(start, end, title);
      });
    });

    // Save Topic button
    document.querySelectorAll('.save-topic-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const tid = btn.getAttribute('data-topic-id');
        const title = btn.getAttribute('data-title') || 'Topic Cut';
        const start = parseFloat(btn.getAttribute('data-start') || '0');
        const end = parseFloat(btn.getAttribute('data-end') || '0');
        const dur = Math.max(1, Math.round(end - start));

        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

        const success = await saveClipToLibrary({
          video_id: state.videoId,
          clip_id: tid,
          title: title,
          start_time: start,
          end_time: end,
          duration: dur,
          video_url: `/api/videos/${state.videoId}/clip/${tid}`,
          download_url: `/api/videos/${state.videoId}/clip/${tid}`
        });

        if (success) {
          btn.innerHTML = '<i class="fa-solid fa-bookmark" style="color:var(--color-green);"></i> Saved';
          btn.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          btn.style.color = 'var(--color-green)';
        } else {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-regular fa-bookmark"></i> Save';
        }
      });
    });

    // Single Clip Generation
    document.querySelectorAll('.generate-single-clip-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const tid = btn.getAttribute('data-topic-id');
        await generateClips([tid]);
      });
    });
  }

  let activePreviewStopHandler = null;

  function previewAtTimestamp(start, end = null, title = '') {
    if (!state.videoId && !state.selectedFile) {
      alert("Please upload or import a video first.");
      return;
    }

    if (outputSection) {
      outputSection.classList.remove('hidden');
    }

    const fullVideoUrl = state.selectedFile 
      ? URL.createObjectURL(state.selectedFile) 
      : `${state.apiBase}/api/videos/${state.videoId}/stream`;

    if (activePreviewStopHandler) {
      finalVideoPlayer.removeEventListener('timeupdate', activePreviewStopHandler);
      activePreviewStopHandler = null;
    }

    const seekAndPlay = () => {
      try {
        finalVideoPlayer.currentTime = Math.max(0, start);
      } catch (err) {
        console.warn("Could not set currentTime immediately:", err);
      }
      const playPromise = finalVideoPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          finalVideoPlayer.muted = true;
          finalVideoPlayer.play().catch(() => {});
        });
      }

      if (end && end > start) {
        activePreviewStopHandler = () => {
          if (finalVideoPlayer.currentTime >= end) {
            finalVideoPlayer.pause();
            finalVideoPlayer.removeEventListener('timeupdate', activePreviewStopHandler);
            activePreviewStopHandler = null;
          }
        };
        finalVideoPlayer.addEventListener('timeupdate', activePreviewStopHandler);
      }
    };

    const isCurrentSource = finalVideoPlayer.src && 
      (finalVideoPlayer.src.includes(`${state.videoId}/stream`) || (state.selectedFile && finalVideoPlayer.src.startsWith('blob:')));

    if (isCurrentSource && finalVideoPlayer.readyState >= 1) {
      seekAndPlay();
    } else {
      finalVideoPlayer.src = fullVideoUrl;
      finalVideoPlayer.addEventListener('loadedmetadata', seekAndPlay, { once: true });
      finalVideoPlayer.load();
    }

    if (nowPlayingLabel) {
      nowPlayingLabel.textContent = title ? `Previewing Scene: ${title} (${formatTime(start)} — ${formatTime(end || start + 15)})` : `Previewing Scene at ${formatTime(start)}`;
    }

    if (tabMergedVideo) tabMergedVideo.classList.remove('active');
    if (tabIndividualClips) tabIndividualClips.classList.remove('active');

    if (outputSection) {
      outputSection.scrollIntoView({ behavior: 'smooth' });
    }
  }

  selectAllTopicsCheckbox.addEventListener('change', (e) => {
    const isChecked = e.target.checked;
    document.querySelectorAll('.topic-checkbox').forEach(cb => {
      cb.checked = isChecked;
      const tid = cb.getAttribute('data-topic-id');
      const card = document.getElementById(`card_${tid}`);
      if (isChecked) {
        state.selectedTopicIds.add(tid);
        if (card) card.classList.add('selected');
      } else {
        state.selectedTopicIds.delete(tid);
        if (card) card.classList.remove('selected');
      }
    });
    updateSelectionCounter();
  });

  function updateSelectionCounter() {
    const count = state.selectedTopicIds.size;
    selectedCountBadge.textContent = `${count} Selected`;
    if (selectAllTopicsCheckbox) {
      selectAllTopicsCheckbox.checked = (count === state.topics.length && count > 0);
    }
  }

  // -------------------------------------------------------------
  // 5. CLIP GENERATION & SELECTIVE MERGE
  // -------------------------------------------------------------
  function updateStudioMetrics(origDuration, repurposedDuration) {
    const orig = origDuration || state.duration || 0;
    const repur = repurposedDuration || 0;

    if (metricOriginalDuration) {
      metricOriginalDuration.textContent = formatTime(orig);
    }
    if (metricRepurposedDuration) {
      metricRepurposedDuration.textContent = formatTime(repur);
    }
    if (metricCompression) {
      const pct = (orig > 0 && repur > 0) ? Math.max(0, Math.min(99, Math.round((1 - (repur / orig)) * 100))) : 0;
      const savedSec = Math.max(0, orig - repur);
      metricCompression.textContent = `${pct}% Context-Preserving Compression`;
      if (metricCompressionDetail) {
        metricCompressionDetail.textContent = `Saved ${formatTime(savedSec)} (${pct}% reduction) with zero syllable clipping & complete boundaries preserved.`;
      }
    }
  }

  function playClipInStudio(clip, idx, clipUrl) {
    state.activeClip = clip;

    if (!clipUrl) {
      const endpoint = clip.video_url || clip.download_url || `/api/videos/${state.videoId}/clip/${clip.topic_id || clip.id}`;
      clipUrl = `${state.apiBase}${endpoint}?t=${Date.now()}`;
    }

    if (outputSection) {
      outputSection.classList.remove('hidden');
    }

    finalVideoPlayer.src = clipUrl;
    finalVideoPlayer.load();

    const subUrl = clip.subtitle_url 
      ? `${state.apiBase}${clip.subtitle_url}` 
      : `${state.apiBase}/api/videos/${state.videoId}/clip/${clip.topic_id || clip.id}/subtitles`;

    if (videoSubtitlesTrack) {
      videoSubtitlesTrack.src = subUrl;
      videoSubtitlesTrack.default = true;
    }
    if (downloadSubtitlesBtn) {
      downloadSubtitlesBtn.href = subUrl;
      downloadSubtitlesBtn.style.display = 'inline-flex';
    }

    const playPromise = finalVideoPlayer.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        // If unmuted autoplay blocked by browser policy, play muted so user sees video
        finalVideoPlayer.muted = true;
        finalVideoPlayer.play().catch(() => {});
      });
    }

    // Highlight active playlist item
    document.querySelectorAll('.studio-clip-item').forEach((item, i) => {
      const btn = item.querySelector('.studio-clip-play-btn');
      if (i === idx) {
        item.classList.add('active-clip');
        if (btn) btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Playing';
      } else {
        item.classList.remove('active-clip');
        if (btn) btn.innerHTML = '<i class="fa-solid fa-play"></i> Play';
      }
    });

    const clipName = clip.text || clip.name || `Topic Cut ${idx + 1}`;
    if (nowPlayingLabel) nowPlayingLabel.textContent = `Active: Clip ${idx + 1} — ${clipName} (${clip.duration}s)`;
    if (downloadVideoBtn) {
      downloadVideoBtn.href = clipUrl;
      downloadVideoBtn.setAttribute('download', clip.filename || `clip_${idx + 1}.mp4`);
    }
    if (downloadVideoBtnText) {
      downloadVideoBtnText.textContent = `Download Clip ${idx + 1} (${clip.duration}s)`;
    }

    updateStudioMetrics(state.duration, clip.duration);

    if (tabIndividualClips) tabIndividualClips.classList.add('active');
    if (tabMergedVideo) tabMergedVideo.classList.remove('active');
    if (outputSection) outputSection.scrollIntoView({ behavior: 'smooth' });
  }

  function renderStudioClipsPlaylist(clips, activeIdx = 0) {
    clipsCountBadge.textContent = clips.length;

    if (studioClipsList) {
      studioClipsList.innerHTML = '';
      if (!clips || clips.length === 0) {
        studioClipsList.innerHTML = `<p class="text-secondary" style="font-size: 0.82rem; padding: 0.5rem 0;">No clips rendered yet. Click "Generate Selected Clips" or "Merge" above.</p>`;
      } else {
        clips.forEach((c, idx) => {
          const clipItem = document.createElement('div');
          clipItem.className = `studio-clip-item ${idx === activeIdx ? 'active-clip' : ''}`;
          clipItem.id = `studioClip_${idx}`;
          const clipUrl = `${state.apiBase}${c.video_url || c.download_url}?t=${Date.now()}`;
          const clipName = c.text || c.name || `Topic Cut ${idx + 1}`;

          clipItem.innerHTML = `
            <div class="studio-clip-info">
              <span class="studio-clip-badge">Clip ${idx + 1}</span>
              <span class="studio-clip-title" title="${escapeHtml(clipName)}">${escapeHtml(clipName)}</span>
              ${c.key_information ? `<span style="display:block; font-size:0.74rem; color:var(--text-secondary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:260px; margin-top:2px;">${escapeHtml(c.key_information)}</span>` : ''}
            </div>
            <div class="studio-clip-meta">
              <span class="studio-clip-duration">${c.duration}s</span>
              <button class="btn-icon-subtle studio-clip-save-btn" data-index="${idx}" title="Save permanently to My Library">
                <i class="fa-regular fa-bookmark"></i>
              </button>
              <button class="studio-clip-play-btn" data-url="${clipUrl}" data-index="${idx}">
                <i class="fa-solid fa-play"></i> ${idx === activeIdx ? 'Playing' : 'Play'}
              </button>
            </div>
          `;

          clipItem.addEventListener('click', (e) => {
            if (e.target.closest('.studio-clip-save-btn')) return;
            playClipInStudio(c, idx, clipUrl);
          });

          studioClipsList.appendChild(clipItem);
        });

        // Bind playlist bookmark button events
        document.querySelectorAll('.studio-clip-save-btn').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.getAttribute('data-index') || '0', 10);
            const clip = clips[idx];
            if (!clip) return;
            const clipName = clip.text || clip.name || `Topic Cut ${idx + 1}`;
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
            const ok = await saveClipToLibrary({
              video_id: state.videoId,
              clip_id: clip.topic_id || clip.id || `clip_${idx}`,
              title: clipName,
              start_time: clip.start_time || 0,
              end_time: clip.end_time || clip.duration,
              duration: clip.duration,
              video_url: clip.video_url || clip.download_url,
              download_url: clip.download_url || clip.video_url
            });
            if (ok) {
              btn.innerHTML = '<i class="fa-solid fa-bookmark" style="color:var(--color-green);"></i>';
            } else {
              btn.innerHTML = '<i class="fa-regular fa-bookmark"></i>';
            }
          });
        });
      }
    }

    if (clipsGrid) {
      clipsGrid.innerHTML = '';
      clips.forEach((c, idx) => {
        const card = document.createElement('div');
        card.className = 'clip-card';
        const clipUrl = `${state.apiBase}${c.video_url || c.download_url}?t=${Date.now()}`;
        const clipName = c.text || c.name || `Topic Cut ${idx + 1}`;
        card.innerHTML = `
          <div class="clip-card-header">
            <div class="clip-badge"><i class="fa-solid fa-film"></i> Clip ${idx + 1}</div>
            <span class="clip-time">${c.duration}s</span>
          </div>
          <div class="clip-card-body">
            <p class="clip-transcript-snip"><strong>${escapeHtml(clipName)}</strong></p>
            ${c.key_information ? `<p style="font-size: 0.8rem; color: var(--text-secondary); margin: 4px 0 8px 0; line-height: 1.35;">${escapeHtml(c.key_information)}</p>` : ''}
            <div class="clip-footer-actions" style="margin-top: 10px; display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap;">
              <button class="btn-ghost btn-sm save-grid-clip-btn" data-index="${idx}">
                <i class="fa-regular fa-bookmark"></i> Save
              </button>
              <a href="${clipUrl}" download="${c.filename}" class="btn-ghost btn-sm">
                <i class="fa-solid fa-download"></i> Download
              </a>
              <button class="btn-primary btn-sm play-clip-btn" data-url="${clipUrl}" data-index="${idx}">
                <i class="fa-solid fa-play"></i> Play in Studio
              </button>
            </div>
          </div>
        `;
        clipsGrid.appendChild(card);
      });

      document.querySelectorAll('.save-grid-clip-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const idx = parseInt(btn.getAttribute('data-index') || '0', 10);
          const clip = clips[idx] || clips[0];
          const clipName = clip.text || clip.name || `Topic Cut ${idx + 1}`;
          btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
          const ok = await saveClipToLibrary({
            video_id: state.videoId,
            clip_id: clip.topic_id || clip.id || `clip_${idx}`,
            title: clipName,
            start_time: clip.start_time || 0,
            end_time: clip.end_time || clip.duration,
            duration: clip.duration,
            video_url: clip.video_url || clip.download_url,
            download_url: clip.download_url || clip.video_url
          });
          if (ok) {
            btn.innerHTML = '<i class="fa-solid fa-bookmark" style="color:var(--color-green);"></i> Saved';
          } else {
            btn.innerHTML = '<i class="fa-regular fa-bookmark"></i> Save';
          }
        });
      });

      document.querySelectorAll('.play-clip-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const idx = parseInt(btn.getAttribute('data-index') || '0', 10);
          const clip = clips[idx] || clips[0];
          playClipInStudio(clip, idx, btn.getAttribute('data-url'));
          outputSection.scrollIntoView({ behavior: 'smooth' });
        });
      });
    }
  }

  generateSelectedClipsBtn.addEventListener('click', async () => {
    let ids = Array.from(state.selectedTopicIds);
    if (ids.length === 0) {
      if (state.topics && state.topics.length > 0) {
        ids = state.topics.map(t => t.id);
      } else {
        alert("Please select at least one topic to generate clips.");
        return;
      }
    }
    await generateClips(ids);
  });

  async function generateClips(topicIds) {
    if (!state.videoId) return;
    setProcessing(true, "Rendering Individual Topic Clips...", `Deterministically rendering ${topicIds.length} clip(s) with FFmpeg (+80ms lead-in, +160ms tail-out)...`, 85, 'boundaries');
    try {
      const res = await fetch(`${state.apiBase}/api/videos/${state.videoId}/generate-clips`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic_ids: topicIds })
      });

      if (!res.ok) {
        const errMsg = await parseErrorResponse(res, "Clip generation failed");
        throw new Error(errMsg);
      }

      const data = await res.json();
      state.clips = data.clips || [];
      renderStudioClipsPlaylist(state.clips, 0);

      // Immediately display and play Clip 1 right in the Studio Player!
      if (state.clips.length > 0) {
        const firstClip = state.clips[0];
        const clipUrl = `${state.apiBase}${firstClip.video_url || firstClip.download_url}?t=${Date.now()}`;
        playClipInStudio(firstClip, 0, clipUrl);
      }

      const totalClipsDur = state.clips.reduce((acc, c) => acc + (c.duration || 0), 0);
      updateStudioMetrics(state.duration, totalClipsDur);

      setProcessing(false);
      outputSection.classList.remove('hidden');
      tabIndividualClips.classList.add('active');
      tabMergedVideo.classList.remove('active');
      outputSection.scrollIntoView({ behavior: 'smooth' });

    } catch (e) {
      alert(`Clip Generation Error: ${e.message}`);
      setProcessing(false);
    }
  }

  mergeSelectedClipsBtn.addEventListener('click', async () => {
    await mergeSelectedClips();
  });

  quickMergeSelectedBtn.addEventListener('click', async () => {
    await mergeSelectedClips();
  });

  if (mergeClipsActionBtn) {
    mergeClipsActionBtn.addEventListener('click', async () => {
      await mergeSelectedClips();
    });
  }

  async function mergeSelectedClips() {
    let ids = Array.from(state.selectedTopicIds);
    if (ids.length === 0) {
      if (state.topics && state.topics.length > 0) {
        ids = state.topics.map(t => t.id);
      } else if (state.clips && state.clips.length > 0) {
        ids = state.clips.map(c => c.topic_id);
      }
    }

    if (!state.videoId) {
      alert("Please upload and index a video first.");
      return;
    }

    if (ids.length === 0) {
      alert("No topics found to merge.");
      return;
    }

    setProcessing(true, "Merging Selected Clips...", `Assembling ${ids.length} verified clip(s) with deterministic FFmpeg concat...`, 90, 'boundaries');
    try {
      const res = await fetch(`${state.apiBase}/api/videos/${state.videoId}/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic_ids: ids })
      });

      if (!res.ok) {
        const errMsg = await parseErrorResponse(res, "Merge failed");
        throw new Error(errMsg);
      }

      const data = await res.json();
      state.mergedVideoUrl = `${state.apiBase}${data.video_url}?t=${Date.now()}`;

      // Refresh clips list in background
      try {
        const clipsRes = await fetch(`${state.apiBase}/api/videos/${state.videoId}/clips`);
        if (clipsRes.ok) {
          const clipsData = await clipsRes.json();
          state.clips = clipsData.clips || [];
          renderStudioClipsPlaylist(state.clips, -1);
        }
      } catch (cErr) {
        console.warn("Clips reload:", cErr);
      }

      // Update player with merged video and play immediately!
      finalVideoPlayer.src = state.mergedVideoUrl;
      finalVideoPlayer.load();
      finalVideoPlayer.play().catch(() => {});

      downloadVideoBtn.href = state.mergedVideoUrl;
      downloadVideoBtnText.textContent = "Download Merged Master (.mp4)";
      downloadVideoBtn.setAttribute('download', `${state.videoId}_merged_master.mp4`);

      // Update metrics with merged duration & compression %
      const totalMergedDur = data.total_duration || state.clips.reduce((acc, c) => acc + (c.duration || 0), 0);
      updateStudioMetrics(state.duration, totalMergedDur);

      nowPlayingLabel.textContent = "Active: Merged Master Video";
      tabMergedVideo.classList.add('active');
      tabIndividualClips.classList.remove('active');

      setProcessing(false);
      outputSection.classList.remove('hidden');
      outputSection.scrollIntoView({ behavior: 'smooth' });

    } catch (e) {
      alert(`Merge Error: ${e.message}`);
      setProcessing(false);
    }
  }

  // -------------------------------------------------------------
  // 6. STUDIO TABS (MERGED MASTER VS INDIVIDUAL CLIPS)
  // -------------------------------------------------------------
  tabMergedVideo.addEventListener('click', () => {
    tabMergedVideo.classList.add('active');
    tabIndividualClips.classList.remove('active');
    if (individualClipsPanel) individualClipsPanel.classList.add('hidden');
    if (studioPlaylistContainer) studioPlaylistContainer.classList.remove('hidden');
    if (state.mergedVideoUrl) {
      finalVideoPlayer.src = state.mergedVideoUrl;
      finalVideoPlayer.load();
      finalVideoPlayer.play().catch(() => {});
      nowPlayingLabel.textContent = "Active: Merged Master Video";
      downloadVideoBtn.href = state.mergedVideoUrl;
      downloadVideoBtnText.textContent = "Download Merged Master (.mp4)";
      downloadVideoBtn.setAttribute('download', `${state.videoId}_merged_master.mp4`);
      const totalClipsDur = state.clips.reduce((acc, c) => acc + (c.duration || 0), 0);
      updateStudioMetrics(state.duration, totalMergedDur || totalClipsDur);
    } else {
      mergeSelectedClips();
    }
  });

  tabIndividualClips.addEventListener('click', () => {
    tabIndividualClips.classList.add('active');
    tabMergedVideo.classList.remove('active');
    if (individualClipsPanel) individualClipsPanel.classList.remove('hidden');
    if (state.clips && state.clips.length > 0) {
      playClipInStudio(state.clips[0], 0, `${state.apiBase}${state.clips[0].video_url || state.clips[0].download_url}?t=${Date.now()}`);
    } else {
      const ids = Array.from(state.selectedTopicIds);
      if (ids.length > 0) {
        generateClips(ids);
      }
    }
  });

  // -------------------------------------------------------------
  // 7. PRESERVED AUDIENCE PROFILES & CONFIG
  // -------------------------------------------------------------
  async function fetchAudiences() {
    try {
      const res = await fetch(`${state.apiBase}/api/audience/profiles`);
      if (res.ok) {
        state.audiences = await res.json();
        renderAudiences(state.audiences);
      }
    } catch (e) {
      console.warn("Could not fetch audience profiles:", e);
    }
  }

  function renderAudiences(audiences) {
    if (!audienceGrid) return;
    audienceGrid.innerHTML = '';
    audiences.forEach((a, idx) => {
      const card = document.createElement('div');
      card.className = `audience-card ${idx === 0 ? 'selected' : ''}`;
      card.innerHTML = `
        <div class="card-icon"><i class="fa-solid fa-users"></i></div>
        <h3>${escapeHtml(a.name)}</h3>
        <p class="target-role">${escapeHtml(a.target_role)}</p>
        <p class="desc">${escapeHtml(a.description)}</p>
        <div class="compression-pill"><i class="fa-solid fa-compress"></i> ${escapeHtml(a.compression_target)}</div>
      `;
      audienceGrid.appendChild(card);
    });
  }

  // -------------------------------------------------------------
  // UTILITIES & PROGRESS HELPERS
  // -------------------------------------------------------------
  function setProcessing(show, title = '', detail = '', pct = 25, stageKey = null) {
    if (show) {
      processingCard.classList.remove('hidden');
      processingTitle.textContent = title;
      processingDetail.textContent = detail;
      progressBar.style.width = `${pct}%`;
      if (stageKey) setStagePill(stageKey);
    } else {
      processingCard.classList.add('hidden');
    }
  }

  function setStagePill(activeKey) {
    Object.keys(stagePills).forEach(k => {
      if (stagePills[k]) {
        if (k === activeKey) stagePills[k].classList.add('active');
        else stagePills[k].classList.remove('active');
      }
    });
  }

  function updateStepIndicator(stepNum) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`stepIndicator${i}`);
      if (el) {
        if (i <= stepNum) el.classList.add('active');
        else el.classList.remove('active');
      }
    }
  }

  function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return "00:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // -------------------------------------------------------------
  // TOAST NOTIFICATIONS & AUTH UTILITIES
  // -------------------------------------------------------------
  function showToast(message, type = 'success', duration = 3500) {
    let container = document.getElementById('toastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toastContainer';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    const iconClass = type === 'success' ? 'fa-circle-check' : (type === 'error' ? 'fa-circle-xmark' : 'fa-circle-info');
    toast.innerHTML = `<i class="fa-solid ${iconClass}"></i> <span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('hide');
      setTimeout(() => toast.remove(), 350);
    }, duration);
  }

  function getAuthHeaders(includeContentType = true) {
    const headers = {};
    if (includeContentType) {
      headers['Content-Type'] = 'application/json';
    }
    if (state.token) {
      headers['Authorization'] = `Bearer ${state.token}`;
    }
    return headers;
  }

  function updateUserUI() {
    if (state.token && state.user) {
      if (authBtn) authBtn.classList.add('hidden');
      if (userProfilePill) userProfilePill.classList.remove('hidden');
      if (userNameLabel) userNameLabel.textContent = state.user.name || 'User';
      if (userAvatarMini) {
        if (state.user.avatar_url) {
          userAvatarMini.innerHTML = `<img src="${escapeHtml(state.user.avatar_url)}" alt="Avatar" style="width:100%;height:100%;border-radius:50%;object-fit:cover;" />`;
        } else {
          const initial = (state.user.name || state.user.phone || 'U').charAt(0).toUpperCase();
          userAvatarMini.textContent = initial;
        }
      }
      refreshLibraryCount();
    } else {
      if (authBtn) authBtn.classList.remove('hidden');
      if (userProfilePill) userProfilePill.classList.add('hidden');
      if (navLibraryCountBadge) navLibraryCountBadge.textContent = '0';
    }
  }

  async function checkAuthState() {
    if (!state.token) {
      updateUserUI();
      return;
    }
    try {
      const res = await fetch(`${state.apiBase}/api/auth/me`, {
        headers: getAuthHeaders(false)
      });
      if (res.ok) {
        const data = await res.json();
        state.user = data.user;
        localStorage.setItem('vidara_user', JSON.stringify(state.user));
        updateUserUI();
      } else {
        // Token expired or invalid
        state.token = null;
        state.user = null;
        localStorage.removeItem('vidara_token');
        localStorage.removeItem('vidara_user');
        updateUserUI();
      }
    } catch (e) {
      console.warn("Could not check auth state:", e);
      updateUserUI();
    }
  }

  async function refreshLibraryCount() {
    if (!state.token) return;
    try {
      const res = await fetch(`${state.apiBase}/api/library/clips`, {
        headers: getAuthHeaders(false)
      });
      if (res.ok) {
        const data = await res.json();
        const count = data.count || 0;
        if (navLibraryCountBadge) navLibraryCountBadge.textContent = count;
        if (libClipsCountBadge) libClipsCountBadge.textContent = count;
      }
    } catch (e) {
      console.warn("Could not refresh library count:", e);
    }
  }

  async function saveClipToLibrary(clipData) {
    if (!state.token) {
      showToast("Please sign in to save clips permanently to your dashboard.", "info", 4000);
      if (authModal) authModal.classList.remove('hidden');
      return false;
    }

    try {
      const payload = {
        video_id: clipData.video_id || state.videoId || 'active',
        clip_id: String(clipData.clip_id),
        title: clipData.title || 'Saved Topic Clip',
        video_title: clipData.video_title || state.videoTitle || 'Video',
        start_time: parseFloat(clipData.start_time) || 0,
        end_time: parseFloat(clipData.end_time) || parseFloat(clipData.duration) || 10,
        duration: parseFloat(clipData.duration) || 10,
        video_url: clipData.video_url || '',
        download_url: clipData.download_url || clipData.video_url || ''
      };

      const res = await fetch(`${state.apiBase}/api/library/clips/save`, {
        method: 'POST',
        headers: getAuthHeaders(true),
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errMsg = await parseErrorResponse(res, "Could not save clip to library");
        throw new Error(errMsg);
      }

      const resData = await res.json();
      showToast(resData.message || `"${payload.title}" saved to your permanent library!`, "success");
      refreshLibraryCount();
      return true;
    } catch (err) {
      showToast(err.message, "error");
      return false;
    }
  }

  // -------------------------------------------------------------
  // AUTHENTICATION MODAL LOGIC (Phone OTP & Google Sign-In)
  // -------------------------------------------------------------
  function initAuthUI() {
    if (authBtn) {
      authBtn.addEventListener('click', () => {
        if (authModal) authModal.classList.remove('hidden');
      });
    }

    if (closeAuthModalBtn) {
      closeAuthModalBtn.addEventListener('click', () => {
        if (authModal) authModal.classList.add('hidden');
      });
    }

    if (authModal) {
      authModal.addEventListener('click', (e) => {
        if (e.target === authModal) authModal.classList.add('hidden');
      });
    }

    if (tabAuthPhone && tabAuthGoogle) {
      tabAuthPhone.addEventListener('click', () => {
        tabAuthPhone.classList.add('active');
        tabAuthGoogle.classList.remove('active');
        if (panelAuthPhone) panelAuthPhone.classList.remove('hidden');
        if (panelAuthGoogle) panelAuthGoogle.classList.add('hidden');
      });

      tabAuthGoogle.addEventListener('click', () => {
        tabAuthGoogle.classList.add('active');
        tabAuthPhone.classList.remove('active');
        if (panelAuthGoogle) panelAuthGoogle.classList.remove('hidden');
        if (panelAuthPhone) panelAuthPhone.classList.add('hidden');
      });
    }

    if (sendOtpBtn && authPhoneInput) {
      sendOtpBtn.addEventListener('click', async () => {
        const phone = authPhoneInput.value.trim();
        if (!phone || phone.length < 7) {
          alert("Please enter a valid phone number (e.g. +91 9876543210).");
          authPhoneInput.focus();
          return;
        }

        sendOtpBtn.disabled = true;
        sendOtpBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

        try {
          const res = await fetch(`${state.apiBase}/api/auth/phone/send-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone })
          });

          if (!res.ok) {
            const err = await parseErrorResponse(res, "Failed to send OTP");
            throw new Error(err);
          }

          const data = await res.json();
          if (otpSection) otpSection.classList.remove('hidden');
          if (devOtpBadge && data.dev_otp) devOtpBadge.textContent = data.dev_otp;
          if (authOtpInput) {
            if (data.dev_otp) authOtpInput.value = data.dev_otp;
            authOtpInput.focus();
          }
          showToast(`Verification code sent to ${phone}!`, "success");
        } catch (e) {
          showToast(e.message, "error");
        } finally {
          sendOtpBtn.disabled = false;
          sendOtpBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Resend OTP';
        }
      });
    }

    if (verifyOtpBtn) {
      verifyOtpBtn.addEventListener('click', async () => {
        const phone = authPhoneInput ? authPhoneInput.value.trim() : '';
        const otp = authOtpInput ? authOtpInput.value.trim() : '';
        const name = authNameInput ? authNameInput.value.trim() : '';

        if (!phone) {
          alert("Phone number is required.");
          return;
        }
        if (!otp || otp.length !== 6) {
          alert("Please enter the 6-digit OTP verification code.");
          authOtpInput && authOtpInput.focus();
          return;
        }

        verifyOtpBtn.disabled = true;
        verifyOtpBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Verifying...';

        try {
          const res = await fetch(`${state.apiBase}/api/auth/phone/verify-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone, otp, name: name || undefined })
          });

          if (!res.ok) {
            const err = await parseErrorResponse(res, "Verification failed");
            throw new Error(err);
          }

          const data = await res.json();
          state.token = data.token;
          state.user = data.user;
          localStorage.setItem('vidara_token', state.token);
          localStorage.setItem('vidara_user', JSON.stringify(state.user));

          if (authModal) authModal.classList.add('hidden');
          updateUserUI();
          showToast(`Welcome to Vidara, ${state.user.name || 'User'}!`, "success");
        } catch (e) {
          showToast(e.message, "error");
        } finally {
          verifyOtpBtn.disabled = false;
          verifyOtpBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Verify & Enter Vidara';
        }
      });
    }

    if (directGoogleLoginBtn) {
      directGoogleLoginBtn.addEventListener('click', async () => {
        const defaultName = authNameInput ? authNameInput.value.trim() : '';
        const userName = defaultName || prompt("Enter your Name for Google Sign-In:", "Vidara User") || "Google User";
        const cleanEmail = userName.toLowerCase().replace(/[^a-z0-9]/g, '') + "@gmail.com";

        directGoogleLoginBtn.disabled = true;
        directGoogleLoginBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Connecting to Google...';

        try {
          const res = await fetch(`${state.apiBase}/api/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              credential: `demo_google:${cleanEmail}:${userName}`,
              name: userName
            })
          });

          if (!res.ok) {
            const err = await parseErrorResponse(res, "Google Sign-In failed");
            throw new Error(err);
          }

          const data = await res.json();
          state.token = data.token;
          state.user = data.user;
          localStorage.setItem('vidara_token', state.token);
          localStorage.setItem('vidara_user', JSON.stringify(state.user));

          if (authModal) authModal.classList.add('hidden');
          updateUserUI();
          showToast(`Signed in with Google as ${state.user.name}!`, "success");
        } catch (e) {
          showToast(e.message, "error");
        } finally {
          directGoogleLoginBtn.disabled = false;
          directGoogleLoginBtn.innerHTML = '<i class="fa-brands fa-google text-red"></i> <span>Continue with Google Account</span>';
        }
      });
    }

    if (logoutBtn) {
      logoutBtn.addEventListener('click', async () => {
        if (!confirm("Are you sure you want to sign out? Your saved clips will remain safely preserved in your dashboard.")) return;
        try {
          await fetch(`${state.apiBase}/api/auth/logout`, { method: 'POST' });
        } catch (_) {}
        state.token = null;
        state.user = null;
        localStorage.removeItem('vidara_token');
        localStorage.removeItem('vidara_user');
        updateUserUI();
        showToast("You have been signed out.", "info");
      });
    }
  }

  // -------------------------------------------------------------
  // PERMANENT LIBRARY & DASHBOARD MODAL
  // -------------------------------------------------------------
  function initLibraryUI() {
    if (navLibraryBtn) {
      navLibraryBtn.addEventListener('click', () => {
        if (!state.token) {
          showToast("Please sign in to access your permanent library.", "info");
          if (authModal) authModal.classList.remove('hidden');
          return;
        }
        if (libraryModal) {
          libraryModal.classList.remove('hidden');
          loadLibraryClips();
          loadLibraryVideos();
        }
      });
    }

    if (closeLibraryModalBtn) {
      closeLibraryModalBtn.addEventListener('click', () => {
        if (libraryModal) libraryModal.classList.add('hidden');
      });
    }

    if (libraryModal) {
      libraryModal.addEventListener('click', (e) => {
        if (e.target === libraryModal) libraryModal.classList.add('hidden');
      });
    }

    if (tabLibClips && tabLibVideos) {
      tabLibClips.addEventListener('click', () => {
        tabLibClips.classList.add('active');
        tabLibVideos.classList.remove('active');
        if (panelLibClips) panelLibClips.classList.remove('hidden');
        if (panelLibVideos) panelLibVideos.classList.add('hidden');
      });

      tabLibVideos.addEventListener('click', () => {
        tabLibVideos.classList.add('active');
        tabLibClips.classList.remove('active');
        if (panelLibVideos) panelLibVideos.classList.remove('hidden');
        if (panelLibClips) panelLibClips.classList.add('hidden');
      });
    }

    if (saveCurrentStudioClipBtn) {
      saveCurrentStudioClipBtn.addEventListener('click', async () => {
        if (!state.videoId) {
          showToast("Please select or import a video first.", "info");
          return;
        }

        const clip = state.activeClip || {
          clip_id: `scene_${Date.now()}`,
          title: state.videoTitle ? `${state.videoTitle} (Highlight)` : 'Featured Moment',
          duration: state.duration || 60,
          start_time: 0,
          end_time: state.duration || 60,
          video_url: finalVideoPlayer.src || `/api/videos/${state.videoId}/stream`
        };

        saveCurrentStudioClipBtn.disabled = true;
        saveCurrentStudioClipBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

        const ok = await saveClipToLibrary({
          video_id: state.videoId,
          clip_id: clip.topic_id || clip.id || clip.clip_id,
          title: clip.text || clip.name || clip.title || 'Saved Moment',
          start_time: clip.start_time || 0,
          end_time: clip.end_time || clip.duration || 60,
          duration: clip.duration || 60,
          video_url: clip.video_url || finalVideoPlayer.src,
          download_url: clip.download_url || clip.video_url || finalVideoPlayer.src
        });

        if (ok) {
          saveCurrentStudioClipBtn.innerHTML = '<i class="fa-solid fa-bookmark" style="color:var(--color-green);"></i> Saved to Library';
        } else {
          saveCurrentStudioClipBtn.disabled = false;
          saveCurrentStudioClipBtn.innerHTML = '<i class="fa-regular fa-bookmark"></i> Save to Library';
        }
      });
    }
  }

  async function loadLibraryClips() {
    if (!state.token || !savedClipsGrid) return;
    savedClipsGrid.innerHTML = '<p class="text-secondary" style="padding: 1rem;"><i class="fa-solid fa-spinner fa-spin"></i> Loading your preserved clips...</p>';

    try {
      const res = await fetch(`${state.apiBase}/api/library/clips`, {
        headers: getAuthHeaders(false)
      });

      if (!res.ok) throw new Error("Could not fetch library clips");

      const data = await res.json();
      const clips = data.clips || [];

      if (libClipsCountBadge) libClipsCountBadge.textContent = clips.length;
      if (navLibraryCountBadge) navLibraryCountBadge.textContent = clips.length;

      if (clips.length === 0) {
        savedClipsGrid.innerHTML = '';
        if (emptyLibraryClipsMsg) emptyLibraryClipsMsg.classList.remove('hidden');
        return;
      }

      if (emptyLibraryClipsMsg) emptyLibraryClipsMsg.classList.add('hidden');
      savedClipsGrid.innerHTML = '';

      clips.forEach((c) => {
        const card = document.createElement('div');
        card.className = 'saved-clip-card';
        card.id = `libCard_${c.id}`;

        const clipUrl = `${state.apiBase}${c.video_url || c.download_url}?t=${Date.now()}`;
        const timeStr = `${formatTime(c.start_time)} — ${formatTime(c.end_time)}`;

        card.innerHTML = `
          <div>
            <div class="saved-clip-header">
              <span class="badge badge-accent" style="font-size:0.75rem;"><i class="fa-solid fa-film"></i> ${escapeHtml(c.video_title || 'Video')}</span>
              <span class="badge badge-neutral" style="font-size:0.75rem;"><i class="fa-regular fa-clock"></i> ${Math.round(c.duration)}s</span>
            </div>
            <h4 class="saved-clip-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</h4>
            <div class="saved-clip-meta">
              <span>Timestamp: ${timeStr}</span>
            </div>
          </div>
          <div class="saved-clip-actions">
            <button class="btn-primary btn-sm lib-play-btn" data-url="${clipUrl}" data-title="${escapeHtml(c.title)}" data-duration="${c.duration}">
              <i class="fa-solid fa-play"></i> Play
            </button>
            <a href="${clipUrl}" download="${c.clip_id}.mp4" class="btn-ghost btn-sm" title="Download MP4">
              <i class="fa-solid fa-download"></i>
            </a>
            <button class="btn-danger-outline btn-sm lib-del-btn" data-id="${c.id}" title="Delete clip permanently">
              <i class="fa-solid fa-trash-can"></i>
            </button>
          </div>
        `;

        savedClipsGrid.appendChild(card);
      });

      // Play button in library
      savedClipsGrid.querySelectorAll('.lib-play-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const url = btn.getAttribute('data-url');
          const title = btn.getAttribute('data-title');
          const dur = parseFloat(btn.getAttribute('data-duration') || '10');

          if (libraryModal) libraryModal.classList.add('hidden');
          playClipInStudio({
            text: title,
            duration: dur,
            video_url: url
          }, 0, url);
        });
      });

      // Delete button in library
      savedClipsGrid.querySelectorAll('.lib-del-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const id = btn.getAttribute('data-id');
          if (!confirm("Are you sure you want to permanently delete this clip from your dashboard?")) return;

          btn.disabled = true;
          btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

          try {
            const delRes = await fetch(`${state.apiBase}/api/library/clips/${id}`, {
              method: 'DELETE',
              headers: getAuthHeaders(false)
            });

            if (!delRes.ok) throw new Error("Could not delete clip");

            const cardEl = document.getElementById(`libCard_${id}`);
            if (cardEl) cardEl.remove();

            refreshLibraryCount();
            showToast("Clip removed from your library.", "info");

            if (savedClipsGrid.children.length === 0 && emptyLibraryClipsMsg) {
              emptyLibraryClipsMsg.classList.remove('hidden');
            }
          } catch (err) {
            showToast(err.message, "error");
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
          }
        });
      });

    } catch (e) {
      savedClipsGrid.innerHTML = `<p class="text-secondary" style="color:#ef4444; padding:1rem;"><i class="fa-solid fa-circle-exclamation"></i> ${e.message}</p>`;
    }
  }

  async function loadLibraryVideos() {
    if (!state.token || !savedVideosList) return;
    savedVideosList.innerHTML = '<p class="text-secondary" style="padding: 1rem;"><i class="fa-solid fa-spinner fa-spin"></i> Loading uploaded projects...</p>';

    try {
      const res = await fetch(`${state.apiBase}/api/library/videos`, {
        headers: getAuthHeaders(false)
      });

      if (!res.ok) throw new Error("Could not fetch video projects");

      const data = await res.json();
      const videos = data.videos || [];

      if (libVideosCountBadge) libVideosCountBadge.textContent = videos.length;

      if (videos.length === 0) {
        savedVideosList.innerHTML = '';
        if (emptyLibraryVideosMsg) emptyLibraryVideosMsg.classList.remove('hidden');
        return;
      }

      if (emptyLibraryVideosMsg) emptyLibraryVideosMsg.classList.add('hidden');
      savedVideosList.innerHTML = '';

      videos.forEach(v => {
        const row = document.createElement('div');
        row.className = 'saved-video-row';
        row.innerHTML = `
          <div>
            <h4 style="font-size:0.92rem; font-weight:600; color:#fff; margin-bottom:4px;">
              <i class="fa-solid fa-file-video icon-accent"></i> ${escapeHtml(v.filename)}
            </h4>
            <div style="font-size:0.78rem; color:var(--text-muted); display:flex; gap:12px;">
              <span>Duration: ${formatTime(v.duration)}</span>
              <span>Size: ${(v.filesize / (1024 * 1024)).toFixed(1)} MB</span>
              <span>Added: ${v.created_at ? v.created_at.slice(0, 10) : 'Recent'}</span>
            </div>
          </div>
          <div>
            <button class="btn-ghost btn-sm load-video-project-btn" data-id="${v.id}" data-filename="${escapeHtml(v.filename)}" data-duration="${v.duration}">
              <i class="fa-solid fa-arrow-up-right-from-square"></i> Open
            </button>
          </div>
        `;
        savedVideosList.appendChild(row);
      });

      savedVideosList.querySelectorAll('.load-video-project-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const vid = btn.getAttribute('data-id');
          const fname = btn.getAttribute('data-filename');
          const dur = parseFloat(btn.getAttribute('data-duration') || '0');

          state.videoId = vid;
          state.duration = dur;
          state.videoTitle = fname;

          if (videoFileName) videoFileName.textContent = fname;
          if (videoFileStats) videoFileStats.textContent = `${formatTime(dur)} • Loaded from Library`;
          if (videoPreviewBar) videoPreviewBar.classList.remove('hidden');

          if (libraryModal) libraryModal.classList.add('hidden');
          showToast(`Loaded "${fname}" into active workspace.`, "success");

          previewAtTimestamp(0, null, fname);
        });
      });

    } catch (e) {
      savedVideosList.innerHTML = `<p class="text-secondary" style="color:#ef4444; padding:1rem;"><i class="fa-solid fa-circle-exclamation"></i> ${e.message}</p>`;
    }
  }

  // -------------------------------------------------------------
  // API KEY CONFIGURATION MODAL (DUAL KEYS + MASTER KEY)
  // -------------------------------------------------------------
  const configKeyBtn = document.getElementById('configKeyBtn');
  const configModal = document.getElementById('configModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const saveConfigBtn = document.getElementById('saveConfigBtn');
  const inputGroqSttKey = document.getElementById('inputGroqSttKey');
  const inputGroqLlmKey = document.getElementById('inputGroqLlmKey');
  const inputGroqKey = document.getElementById('inputGroqKey');

  async function loadCurrentKeys() {
    try {
      const res = await fetch(`${state.apiBase}/api/settings/keys`);
      if (res.ok) {
        const data = await res.json();
        if (data.masked_stt && inputGroqSttKey) inputGroqSttKey.placeholder = `Current: ${data.masked_stt}`;
        if (data.masked_llm && inputGroqLlmKey) inputGroqLlmKey.placeholder = `Current: ${data.masked_llm}`;
        if (data.masked_master && inputGroqKey) inputGroqKey.placeholder = `Current: ${data.masked_master}`;
      }
    } catch (e) {
      console.warn("Could not load key status:", e);
    }
  }

  if (configKeyBtn) {
    configKeyBtn.addEventListener('click', () => {
      if (configModal) {
        configModal.classList.remove('hidden');
        loadCurrentKeys();
      }
    });
  }

  if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => {
      if (configModal) configModal.classList.add('hidden');
    });
  }

  if (saveConfigBtn) {
    saveConfigBtn.addEventListener('click', async () => {
      const sttKey = inputGroqSttKey ? inputGroqSttKey.value.trim() : '';
      const llmKey = inputGroqLlmKey ? inputGroqLlmKey.value.trim() : '';
      const masterKey = inputGroqKey ? inputGroqKey.value.trim() : '';

      if (!sttKey && !llmKey && !masterKey) {
        alert("Please enter at least one Groq API key.");
        return;
      }

      saveConfigBtn.disabled = true;
      saveConfigBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

      try {
        const res = await fetch(`${state.apiBase}/api/settings/keys`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            groq_stt_key: sttKey || undefined,
            groq_llm_key: llmKey || undefined,
            groq_api_key: masterKey || undefined
          })
        });

        const data = await res.json();
        if (res.ok) {
          alert("Groq credentials saved successfully and applied immediately!");
          if (configModal) configModal.classList.add('hidden');
          if (inputGroqSttKey) inputGroqSttKey.value = '';
          if (inputGroqLlmKey) inputGroqLlmKey.value = '';
          if (inputGroqKey) inputGroqKey.value = '';
        } else {
          alert(`Error saving credentials: ${data.detail || 'Unknown error'}`);
        }
      } catch (e) {
        alert(`Failed to save keys: ${e.message}`);
      } finally {
        saveConfigBtn.disabled = false;
        saveConfigBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save & Apply Credentials';
      }
    });
  }
});
