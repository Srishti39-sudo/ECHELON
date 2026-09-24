# Sample side-scan imagery

Two genuine side-scan sonar records and ten tiles cut from one of them, for
demonstrating the survey pipeline on real imagery rather than synthetic noise.

Both are side-scan waterfall records, not forward-looking fan frames. The
distinction matters: SIH26057 is a side-scan problem, and a sonar-literate
reader recognises the difference immediately.

## What is here

| File | Size | Source | Licence |
|---|---|---|---|
| `sidescan-waterfall-strip.jpg` | 3600 x 2758 | Wikimedia Commons, "A side scan waterfall display from the Dragon Prince deep tow fish" | CC BY 4.0 |
| `sidescan-s7-submarine.jpg` | 1676 x 871 | Wikimedia Commons, "Sidescansonarbild 200 khz 2 x 65 m", the wreck of the submarine S-7 | CC BY-SA 4.0 |
| `tiles/` | 10 x 640 | cut from the waterfall strip | as above |

## What was done to them

`sidescan-s7-submarine.jpg` is unmodified. It is the cleaner record of the two:
greyscale, port and starboard channels either side of the water column, a wreck
with a textbook acoustic shadow on the port channel.

`sidescan-waterfall-strip.jpg` had 550 rows of acquisition-software chrome
cropped from the top, detected by row brightness rather than by eye. Nothing
else was altered. It is a photograph of a display, so it carries the display's
false-colour palette and some screen glare, and the original nav readout in the
cropped band is not reproduced anywhere. **Do not treat the strip as
georeferenced.** No navigation file accompanies it, so any survey built from it
is in relative coordinates.

The ten tiles are 640 pixels at stride 512, named `{strip}_{x}_{y}.jpg` where
x and y are pixel offsets into the strip. Of the 30 candidate tiles the ten with
the highest contrast were kept, so the sample is seabed and structure rather
than ten crops of empty water column.

## What the models actually find, measured

Both checkpoints, run over this imagery at their default threshold:

```
sidescan-s7-submarine.jpg   known    ship        0.829
sidescan-s7-submarine.jpg   anomaly  shipwreck   0.431

tiles/..._2048_0.jpg        anomaly  shipwreck   0.486
tiles/..._2048_512.jpg      known    human       0.300
tiles/..._2560_1024.jpg     known    aircraft    0.531
tiles/..._2560_1024.jpg     known    human       0.482
```

The submarine is a true positive and a good one: both models find the same
object, `known.pt` names it confidently, and it is a real wreck on a real
record.

The four tile detections are not. Tile `2560_1024` contains seabed texture, the
dark nadir boundary and a small software annotation, and no identifiable object
at all, yet `known.pt` reports an aircraft at 0.531 and a human at 0.482 on the
shadow edge. The `shipwreck` box on `2048_0` is a tall sliver along the water
column. The failure mode is consistent: **false positives cluster on the
nadir and shadow boundaries**, which is exactly where a side-scan record has
strong contrast and no object.

This is why the per-class confidence floors exist. Applied, the two human calls
and the aircraft call are all withheld and reported as unidentified, and the
submarine survives untouched.
