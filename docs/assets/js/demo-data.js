window.DEMO_DATA = {
  datasets: {
    clean: {
      label: "LibriSpeech test-clean",
      folder: "libri-clean",
      summary: {
        condition: "Clean read speech",
        samples: "4 placeholders",
        bitrate: "Comparable 0.8-1.5 kbps codecs",
        rate: "16 kHz target playback",
      },
      samples: [
        {
          id: "clean_001",
          title: "test-clean sample 1",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for clean_001.",
        },
        {
          id: "clean_002",
          title: "test-clean sample 2",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for clean_002.",
        },
        {
          id: "clean_003",
          title: "test-clean sample 3",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for clean_003.",
        },
        {
          id: "clean_004",
          title: "test-clean sample 4",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for clean_004.",
        },
      ],
    },
    other: {
      label: "LibriSpeech test-other",
      folder: "libri-other",
      summary: {
        condition: "Noisier read speech",
        samples: "4 placeholders",
        bitrate: "Comparable 0.8-1.5 kbps codecs",
        rate: "16 kHz target playback",
      },
      samples: [
        {
          id: "other_001",
          title: "test-other sample 1",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for other_001.",
        },
        {
          id: "other_002",
          title: "test-other sample 2",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for other_002.",
        },
        {
          id: "other_003",
          title: "test-other sample 3",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for other_003.",
        },
        {
          id: "other_004",
          title: "test-other sample 4",
          speaker: "Speaker / chapter / utterance",
          transcript: "Replace this line with the LibriSpeech transcript for other_004.",
        },
      ],
    },
  },
  methods: [
    {
      key: "ground_truth",
      label: "Ground Truth",
      details: "Reference waveform",
    },
    {
      key: "encodec",
      label: "EnCodec",
      details: "24 kHz, 1.5 kbps",
    },
    {
      key: "dac",
      label: "DAC",
      details: "16 kHz, 1.5 kbps",
    },
    {
      key: "speechtokenizer",
      label: "SpeechTokenizer",
      details: "16 kHz, 1.0 kbps",
    },
    {
      key: "bigcodec",
      label: "BigCodec",
      details: "16 kHz, 1.04 kbps",
    },
    {
      key: "mimi",
      label: "Mimi",
      details: "24 kHz, 1.1 kbps",
    },
    {
      key: "xcodec",
      label: "XCodec",
      details: "16 kHz, 1.0 kbps",
    },
    {
      key: "xcodec2",
      label: "XCodec2.0",
      details: "16 kHz, 0.8 kbps",
    },
    {
      key: "dualcodec",
      label: "DualCodec",
      details: "24 kHz, 1.075 kbps",
    },
    {
      key: "xy_tokenizer",
      label: "XY-Tokenizer",
      details: "16 kHz, 1.0 kbps",
    },
    {
      key: "simwhisper_codec",
      label: "SimWhisper-Codec",
      details: "16 kHz, 1.1 kbps",
    },
    {
      key: "bimtokenizer_whisper",
      label: "BiMTokenizer-Whisper",
      details: "Ours, 16 kHz, 1.1 kbps",
      ours: true,
    },
    {
      key: "bimtokenizer_sensevoice",
      label: "BiMTokenizer-SenseVoice",
      details: "Ours, 16 kHz, 1.1 kbps",
      ours: true,
    },
  ],
};
