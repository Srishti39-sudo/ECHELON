/**
 * Every string the operator reads.
 *
 * No component contains user-facing text of its own. Reword the product here.
 * The tone to keep: plain, direct, no exclamation marks, no emoji, nothing that
 * sounds pleased with itself. This is read on a working deck.
 */

export const copy = {
  app: {
    name: 'DeepEcho',
    subtitle: 'Side-scan sonar decision support',
  },

  status: {
    checking: 'Connecting to the assistant',
    ready: 'Ready',
    degraded: 'Index unavailable',
    offline: 'Backend unreachable',
    corpusSummary: (documents: number, chunks: number) =>
      `${documents} reference documents, ${chunks} indexed passages`,
    providerSummary: (provider: string, model: string) => `${provider} / ${model}`,
    detectorStub: 'Detector is a stub. Uploaded tiles return a placeholder detection, not a real one.',
    detectorModels: (models: string[]) =>
      models.length === 0 ? '' : `detector: ${models.join(' and ')}`,
  },

  empty: {
    title: 'Ask about a sonar contact',
    body:
      'Every answer is drawn from a fixed set of published marine references and cites them. ' +
      'Where those references do not cover something, the assistant says so rather than filling the gap.',
    examplesLabel: 'Try',
    examples: [
      'Who do I report a suspected mine to?',
      'The classifier could not identify this contact. What do I do?',
      'How do I tell a manufactured object from a rock on side scan?',
      'Write me an incident report for this detection.',
    ],
  },

  composer: {
    placeholder: 'Ask about this contact',
    placeholderStreaming: 'Answering',
    send: 'Send',
    stop: 'Stop',
    hint: 'Enter to send, Shift and Enter for a new line',
  },

  roles: {
    user: 'You',
    assistant: 'Assistant',
  },

  badge: {
    confidence: (value: number) => `Classifier confidence ${Math.round(value * 100)}%`,
    severityLabel: 'Risk',
    severity: {
      high: 'High',
      medium: 'Medium',
      low: 'Low',
      unknown: 'Unknown',
    },
    intent: {
      question: 'Question',
      explain: 'Detection brief',
      anomaly: 'Unidentified object',
      report: 'Incident report',
    },
    anomaly: 'Unknown object, treated as an anomaly',
    unclassified: 'Unclassified',
  },

  notice: {
    ungroundedTitle: 'Unverified',
    ungroundedBody:
      'No matching reference was found for this answer. Confirm it manually before acting on it.',
    refusalTitle: 'Partly outside the references',
    refusalBody:
      'Part of this answer is the assistant declining to fill a gap the references do not cover. ' +
      'The missing detail is not an omission, it is unavailable.',
    anomalyTitle: 'Object is unidentified',
    anomalyBody:
      'Nothing below identifies this object. Similar known objects are listed as possibilities only, ' +
      'and it stays unidentified until a qualified expert rules.',
    coverageGapTitle: 'No reference covers this class',
    coverageGapBody:
      'The detector named a class the reference documents do not describe. The assistant will say so ' +
      'rather than answering from a document about a different kind of object. Treat what follows as ' +
      'the generic unidentified-object handling, not as guidance for this class.',
    stubDetectionTitle: 'Placeholder detection',
    stubDetectionBody:
      'This record came from the stub detector, not a trained model. It is synthetic and is not evidence.',
  },

  survey: {
    from: 'From the survey hazard map',
    hotspot: 'Hotspot',
    action: 'Recommended action',
    // The map's severity, shown as the map's. Labelled so nobody reads it as
    // something the assistant worked out.
    severityFromMap: 'Survey risk',
    rank: 'Priority rank',
    detections: 'Detections in cell',
    position: 'Centroid',
    notGeoreferenced: 'Not georeferenced. Positions are pixel offsets in the sonar strip, not GPS.',
    demo: 'Synthetic survey data',
    evidence: 'Evidence tile',
  },

  ghosttrace: {
    from: 'From the GhostTrace rescue queue',
    attachedLabel: 'GhostTrace target',
    // Shown on every GhostTrace card and chip when the survey is synthetic. It
    // is not decoration: nothing in a synthetic run is evidence of a real net.
    synthetic: 'Synthetic',
    syntheticBody: 'Synthetic survey data. Nothing here is evidence of a real net.',
    target: 'Target',
    survey: 'Survey',
    objectClass: 'Class',
    confidence: 'Confidence',
    position: 'Position',
    notGeoreferenced: 'Not georeferenced',
    priority: 'Priority',
    priorityValue: (tier: string, rank: string, score: string) =>
      `${tier} · rank ${rank} · score ${score}`,
    // The rescue queue's priority, shown as the queue's. Labelled so nobody
    // reads it as something the assistant worked out.
    priorityChip: 'GhostTrace priority',
    activity: 'Water-column activity',
    habitat: 'Nearest habitat',
    habitatValue: (name: string, distance: string) => `${name}, ${distance} m`,
    propeller: 'Propeller hazard',
    change: 'Change',
    authorities: 'Authorities GhostTrace lists',
    caveats: 'Caveats',
    notAvailable: 'not available',
    dataNotSource:
      'GhostTrace numbers are survey data and heuristics, not reference sources. The assistant attributes them to GhostTrace; authorities and procedures still come from the cited documents.',
    defaultQuestion:
      'Why is this net ranked where it is, and who do the sources say should be told?',
  },

  copilot: {
    modeLabel: 'Answer from',
    modes: {
      auto: 'Auto',
      copilot: 'Mission Copilot',
      reference: 'References only',
    },
    modeHint: {
      auto: 'Questions about the processed surveys go to the Mission Copilot; everything else is answered from the references.',
      copilot: 'Looks up the processed surveys with read-only data tools, then answers with survey records [D] and references [S].',
      reference: 'Answers from the reference documents only, never from survey data.',
    },
    examplesLabel: 'Ask the Mission Copilot',
    examples: [
      'Which net should we recover first across all surveys, and who do we notify?',
      'What changed between the two Mannar surveys?',
      'Which contacts were filtered as false positives and why?',
      'Summarise survey waterfall-strip for the Coast Guard',
    ],
    referenceExamplesLabel: 'Ask the references',
    autoRouted: 'auto-routed',
    consulted: 'Data consulted',
    plannedBy: {
      model: 'planned by the model',
      json_plan: 'planned by the model (JSON)',
      keywords: 'planned from keywords',
    } as Record<string, string>,
    lookingUp: 'Looking up survey data',
    dataOnlyBody:
      'No language model could be reached. What follows is the survey records the copilot looked up, tabled, and short extracts of the references. Nothing in it was generated.',
    dataCount: (n: number) => (n === 1 ? '1 data record' : `${n} data records`),
  },

  language: {
    label: 'Answer language',
    note: 'Source extracts stay in English.',
  },

  offline: {
    badge: 'Offline — sources only, no generated answer',
    body:
      'No language model could be reached. What follows is the retrieved passages, quoted, and the facts handed over with the question. Nothing in it was generated.',
    why: 'Why',
  },

  numbers: {
    title: 'Figures not found in the sources',
    body: (figures: string[]) =>
      `These figures appear in no retrieved passage, in your message or in the handed-over context: ${figures.join(', ')}. Treat them as unverified.`,
  },

  matches: {
    title: 'Nearest known objects',
    caveat:
      'A similarity ranking, not a probability and not an identification. A higher score does not make an identity more likely to be true.',
    similarity: 'Similarity',
    confirms: 'Would confirm',
    rulesOut: 'Would rule out',
    hazard: 'Hazard class',
  },

  data: {
    panelTitle: 'Survey data record',
    tabSources: 'Sources',
    tabData: 'Data',
    listTitle: 'Data records',
    survey: 'Survey',
    file: 'From',
    record: 'Record',
    kind: 'Kind',
    synthetic: 'Synthetic survey data, not evidence of a real object',
    fields: 'Record as the answer saw it',
    openGhosttrace: 'Open in GhostTrace',
    openMap: 'Open on the hazard map',
    markerTitle: (n: number) => `Open data record ${n}`,
    note: 'Survey data from the processed files, not a reference source.',
  },

  citations: {
    panelTitle: 'Source',
    listTitle: 'Sources',
    close: 'Close',
    authority: 'Published by',
    section: 'Section',
    status: 'Status',
    similarity: 'Retrieval score',
    openPdf: 'Open the original publication',
    noPdf: 'No original file is attached to this document',
    markerTitle: (n: number) => `Open source ${n}`,
    count: (n: number) => (n === 1 ? '1 source' : `${n} sources`),
  },

  upload: {
    button: 'Upload sonar tile',
    uploading: 'Reading tile',
    disabled: 'Tile upload is switched off',
    rejected: 'That file type is not accepted',
    detectionsFound: (n: number) => (n === 1 ? '1 contact found' : `${n} contacts found`),
    // Not a failure, and it should not read as one. A whole survey strip finds
    // nothing here because the detector works at 640 pixels: scale a 3600-pixel
    // strip down to that and every contact is a few pixels across. Strips go
    // through the survey pipeline; this path is for a tile.
    noDetections:
      'Nothing detected in that image. If it is a whole survey strip, use the Survey Hazard Map instead: the detector reads one tile at a time and a full strip scales down past the point where anything is visible.',
    attached: 'Attached to this conversation',
    // Sent automatically after an upload, so a tile produces a briefing without
    // the operator having to think of a question first.
    autoBrief: 'Brief me on this contact.',
    contactsLabel: 'Contacts in this tile',
    contact: (n: number, cls: string, confidence: number) =>
      `${n}. ${cls} ${Math.round(confidence * 100)}%`,
    downgraded: 'Class withheld, reported as unidentified',
    detach: 'Remove',
  },

  detection: {
    title: 'Detection on screen',
    none: 'No detection attached',
    fields: {
      object_class: 'Class',
      confidence: 'Confidence',
      bbox: 'Bounding box',
      depth_m: 'Depth (m)',
      latitude: 'Latitude',
      longitude: 'Longitude',
      timestamp: 'Time',
      sensor: 'Sensor',
      platform: 'Platform',
      notes: 'Notes',
      visual_description: 'Description',
      detector_model: 'Detected by',
      detector_class: 'Detector class',
      second_opinion: 'Second opinion',
      downgraded_from: 'Class withheld',
    } as Record<string, string>,
  },

  error: {
    title: 'Request failed',
    backendDown:
      'The assistant backend did not respond. Check that it is running, then send the message again.',
    generic: 'Something went wrong while answering. Nothing above was changed.',
    streamInterrupted:
      'The answer stopped partway. What arrived is shown above and is still cited; the rest is missing.',
    retry: 'Try again',
  },
} as const
