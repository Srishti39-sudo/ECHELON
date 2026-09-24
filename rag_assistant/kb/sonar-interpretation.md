---
title: Reading Side Scan Sonar Contacts
authority: NOAA Office of Coast Survey; NOAA National Ocean Service
source_url: https://nauticalcharts.noaa.gov/publications/docs/standards-and-requirements/specs/HSSD_2021.pdf
source_file: sources/noaa-hssd-2021.pdf; sources/noaa-what-is-sonar.txt
doc_type: technique specification
status: verified
retrieved: 2026-09-10
---

## What sonar returns and what it does not

Sonar is short for Sound Navigation and Ranging. NOAA uses it to develop
nautical charts, locate underwater hazards to navigation, search for and map
objects on the seafloor such as shipwrecks, and map the seafloor itself.

An active sonar transducer emits an acoustic pulse into the water. If an object
lies in the path of that pulse, the sound bounces off it and returns an echo to
the transducer, which measures the strength of the returned signal and the time
between transmission and reception.

A sonar contact is a measurement of returned sound. It is not an identification
of the object that returned it. Identification requires corroboration.

## Detection capability, and what it implies about small objects

NOAA specifies that a side scan sonar system shall be operated so that it is
capable of detecting an object on the sea floor measuring 1 m by 1 m by 1 m from
shadow length measurements (HSSD 2021, section 6.1.2.1). The tow speed shall be
such that an object of that size would be independently ensonified a minimum of
three times per pass (section 6.1.2.2).

Object detection coverage is specified for features of 1 m by 1 m by 1 m and
greater. Complete coverage is specified for features of 2 m and greater.

An object smaller than the detection specification may return nothing at all.
Absence of a contact is not evidence that the seabed is clear.

## Shadow carries the shape

Height is derived from shadow length, not from the return itself. Towfish height
above the bottom shall be 8 to 20 percent of the range scale in use. Below 8
percent, the effective scanning range is defined as 12.5 times the towfish
height, provided adequate echoes have been received.

When towfish height exceeds the maximum threshold, NOAA requires the hydrographer
either to take extra care examining the data for contacts with reduced shadow
lengths, or to re-acquire the data at an appropriate depth. Geometry outside
these bounds degrades height estimation, which is the same measurement a
classifier depends on.

## Which contacts get followed up

NOAA picks contacts with computed target heights rising above the bottom by at
least 5 percent of the depth. Other contacts may be picked if the sonargram
signature, meaning size, shape or pattern qualities, is notable.

All contacts identified shall be developed with a multibeam echo sounder to
determine the least depth of the contact. The least depth measurement should be
determined from a beam within 30 degrees of nadir unless multiple passes were
made over the contact.

The operational point for an automated classifier is that a single side scan
pass is the beginning of the process and not the end of it. A contact is
confirmed by a second sensor or a second geometry, and until then it remains a
contact.

## Conditions that degrade a classification

Excessive bathymetry variability, towfish height outside the specified band,
speed too high to ensonify an object three times per pass, and a cluttered
seabed all reduce the reliability of anything derived from the imagery. Under
these conditions a low-confidence classification is downgraded to unidentified
rather than accepted.
