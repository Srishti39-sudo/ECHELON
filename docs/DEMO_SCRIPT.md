# DeepEcho: 7-Minute Demo Script

*Presenter script with timings. Bold lines are spoken. Italic lines are actions on screen.*

---

## Before you start

| Check | Detail |
|---|---|
| Backend | Started from your own terminal, not from any other tool. Leave that window open. `cd ~/ECHELON && DEEPECHO_ENABLE_UPLOAD=1 .venv/bin/python -m uvicorn backend.app.main:app --port 8000` |
| Frontend | `cd ~/ECHELON/frontend && npm run dev`, open http://localhost:5173 |
| Provider | Groq. The Assistant header must read "groq / provider default" |
| Internet | On, for map tiles and the LLM |
| Tabs open in advance | Live Feed, Live survey, Survey report for `usgs-cat-island-ds563`, GhostTrace on `demo-ghosttrace-mannar-repeat` |
| Files ready | `~/Downloads/xtf-demo/synthetic_A_east_5Hz.xtf` and `~/models/demo/demo_images/shipwreck_2.jpg` in an open Finder window |
| Warm-up | Upload one tile and ask the assistant one question before the judges arrive, so caches are warm |

---

## 0:00 to 0:45. The problem

*Do not touch the screen yet. Face the judges.*

**"A survey vessel tows a side-scan sonar and records the seabed as a long acoustic strip. Somewhere in it may be a wreck, a mine, a pipeline, or a lost fishing net that keeps killing for years. Today an analyst scrolls through kilometres of imagery by eye."**

**"DeepEcho reads the strip, finds the objects, checks each one against the physics of how sound bounces, puts it on the earth, ranks what to deal with first, and answers the operator's questions from published references with citations."**

**"We kept three things separate on purpose: what is there, how bad it is, and what to do about it. Each has different evidence, and a confident sentence is never allowed to raise a severity."**

---

## 0:45 to 1:45. One tile through the detector

*Live Feed. Drop `shipwreck_2.jpg`. It answers in about a second.*

**"This is our own model. One YOLO11 detector, six classes: shipwreck, aircraft, human, pipeline, fishing gear, mine-like object. Trained on six public sonar datasets, mAP50 0.61 on a held-out split it never saw. It replaced two stand-in models that could not see pipelines, mines or fishing gear at all."**

*Point at the two confidence figures.*

**"Two confidences, always. The model said 0.88. Then a verifier looks at the pixels for an acoustic shadow behind the object, a bright face, sharp edges, and fuses that evidence in. That is the 93 percent."**

**"Severity is High because a wreck is a navigation hazard by class. Confidence tells you whether to believe the label. Severity tells you what the label means. We never read severity out of the model's confidence."**

---

## 1:45 to 3:00. A geotagged survey

*Live survey. Upload `synthetic_A_east_5Hz.xtf`. Talk while the stages tick.*

**"Now a raw sonar log, the format real Klein and EdgeTech sonars write. The pipeline reads every ping with its GPS, heading and altitude, tiles the strip with overlap so nothing is cut in half, runs the detector, merges duplicates across the overlap, runs the shadow check, and geotags every contact."**

*When it finishes, point at the summary tiles, then the detections table.*

**"Geo-referenced. Every contact has a latitude, a longitude and a size in metres, computed from slant range, altitude and heading on the WGS-84 ellipsoid."**

**"This file is synthetic, generated with the same library real sonars use, with three objects planted at known positions. Our self-test recovers them to within ten centimetres, so the geometry is proven."**

**"Where we have no navigation, the export says relative coordinates in pixels. We never invent a position."**

*Click the Map tab briefly.*

**"The towfish track and the contacts, at sea, twenty kilometres west of Mumbai."**

---

## 3:00 to 4:00. Real imagery, and the false-positive veto

*Switch to the Survey report tab for `usgs-cat-island-ds563`.*

**"This is a real USGS side-scan mosaic of Cat Island, Mississippi, one metre per pixel, 145 megapixels, 558 tiles."**

**"The detector called five contacts. Four of them are survey-line crossings that look like structure to a neural network. The shadow physics said no shadow behind them, and the system suppressed them."**

*Tick "Show filtered false positives".*

**"It did not delete them. They are kept, marked, with the reason in words. One genuine 22 metre target survived. That is the difference between a model and a system."**

---

## 4:00 to 5:30. GhostTrace

*Switch to the GhostTrace tab, survey `demo-ghosttrace-mannar-repeat`.*

**"For fishing gear specifically, detection is not enough. A recovery team needs to know which net to pull first. GhostTrace answers seven questions for every net."**

*Count them on your fingers.*

**"Is it still catching, from fish echoes in the water column beside it. What habitat is near, from UN and OpenStreetMap layers. Where it would drift, from a particle simulation over ocean currents. Who it could hurt, as a propeller hazard. Has it moved since the last survey. In what order a boat should recover them. And who to tell."**

*Point at the map.*

**"Same line, surveyed three days apart. This net moved forty metres. This one is new. This one is rated a propeller hazard because it sits in eight metres of water."**

*Click the first net's card, then the Alert tab.*

**"The alert names an authority only if a document in our corpus names it for that situation, checked by string match at run time. No phone number, because the sources have none, and it says so. An honest gap sends the operator to find the right desk. An invented one sends the report to the wrong one."**

*Point at the amber banner.*

**"Everything on this page is labelled synthetic, on the card, in the file, in the alert. The method is real. The sea is not."**

---

## 5:30 to 6:30. The assistant

*Click "Ask the assistant" on the net. The starter question sends itself; the answer arrives in two to three seconds.*

**"This is retrieval-augmented generation over ten curated documents, with the original PDFs served alongside. Every claim carries a citation you can open to the passage."**

*Click one citation chip so the source panel opens.*

**"The model is forbidden to compute a number, invent a standoff distance, or name an authority the source does not connect to that hazard. When the corpus is silent it says so and gives the universal safe fallback: do not approach, do not touch, report."**

*Point at the grey block above the answer.*

**"GhostTrace's numbers are shown as data, attributed to GhostTrace. Procedures and authorities come only from the cited documents. The assistant never re-scores anything."**

*Type one question in the box and send it:* `what is the propeller risk here`

**"And it stays about this net for every follow-up."**

---

## 6:30 to 7:00. Close

*Back to the judges.*

**"Everything runs on this laptop. SQLite for uploads, one folder of JSON, CSV and GeoJSON per survey, a local FAISS index, and one hosted LLM call. Fifty-three mechanical evaluation cases and two hundred and eighty engine checks pass. No LangChain, no LangGraph: every grounding rule is plain code we can point at."**

**"Honest limits. Our fishing-gear class is trained on crab pots, because no labelled ghost-net sonar data exists publicly. The first thing we do with field access is collect that and fine-tune. Habitat and current layers cover two Indian coasts today; the fetcher adds any other in one command."**

**"What is there. How bad. What to do. Three answers, three kinds of evidence, never blurred. Thank you."**

---

## Cannot miss

If time runs short, these ten lines must still be said.

1. Our own trained model, six classes, 0.61 mAP50 on held-out data.
2. Two confidences on every contact: the model's score and the physics-fused score.
3. Severity comes from the class, never from confidence.
4. A raw log gives real coordinates; no navigation means no invented position.
5. On the real USGS mosaic, four false positives were suppressed by shadow physics, kept and labelled.
6. GhostTrace: still fishing, habitat, drift, propeller, moved 40 m, route, alert.
7. The alert never invents a contact.
8. The assistant cites every claim and refuses to invent numbers.
9. Synthetic data is labelled synthetic every time it appears.
10. Fishing gear is trained on crab pots, not nets, and here is what we would do about it.

---

## If something goes wrong

| Symptom | Do this |
|---|---|
| Survey fails with "worker exited without reporting a result" | The backend was not started from your terminal. Ctrl+C it, run the command in the checklist, upload again |
| Assistant sits on "Answering" for more than ten seconds | Provider is not Groq. Check the header. Fix `.env`, restart the backend |
| Map is plain blue | It is open sea. Say so, point at the scale bar, zoom out one step |
| No internet | Skip the map tiles and the assistant; everything else is local. Show the USGS report and GhostTrace, which are pre-computed |
| A judge asks about the two blue "provisional" markers during a run | "Single detector calls per tile, before merging. The final list replaces them when the export is written." |

---

## Questions to expect, one line each

- **Is the detector yours?** Yes. Trained by us on public data, weights in the repo, results file in the kit.
- **Why not one confidence?** Because the model's score and the image evidence are different kinds of proof, and hiding one would hide a disagreement.
- **Why is a 38 percent mine-like object High?** Consequence, not probability. The tier says what it would mean if true; the percentage says how much to believe it.
- **Is GhostTrace machine learning?** No. Median and MAD statistics, connected components, a particle simulation, a weighted sum. Every number is recomputable.
- **Is the assistant a chatbot?** It is retrieval with citations and hard refusal rules, evaluated by fifty-three mechanical checks.
- **How would this run on a boat?** ONNX export, no torch, the edge Docker profile; maps and reports are self-contained files.
- **What would you do first with real data?** Label real ghost nets and fine-tune the gear class.
