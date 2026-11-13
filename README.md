# Pediatric gait visualisation tool

A web app that turns a video of a child walking into a clear visualization of the gait patterns for clinicians and parents.

## Overview

This is an independent project that I started on my own initiative. I work as a data analyst at a children's healthcare centre, and I like to look beyond my regular tasks and explore how new technologies could improve the way the centre assesses and supports children.

### The problem

Talking to clinicians and the centre director, I found out that the gait analysis is a big part of their job, as they mainly work with children with musculoskeletal and motor impairments. And what I saw and heard in the centre is that often during the gait diagnostics parents struggle to fully understand what clinician is trying to explain. The difficulty is not only the medical terminologies, but also the gait patterns that are not easy to notice. Without professional training it is hard to pick out what matters when you watch a child walk: how the arms swing, how the legs move, how the body shifts from side to side. When a clinician describes a pattern, parents often have no clear way to find it in what they are looking at, so they struggle to follow the explanation.

### The idea

That's why I decided to develop a web app that will help to solve this problem. Clinicians open the app, upload the video of a child walking and get side-by-side visualization with skeleton and trails.

The tool tracks the keypoints that matter for gait (arms, torso, legs and feet) and shows the walking as a simplified skeleton, so the movement itself is easier to see and the pattern is clear. Parents can see exactly what the clinician is talking about, and the explanation becomes much easier to follow.

The output looks like this:

- **Colour-coded sides:** the child's left side is blue and the right side is orange.
- **Movement trails:** the hands and feet leave a trail behind them, so the path and amount of movement of each limb can be seen at a glance.
- **Skeleton only:** a skeleton walking on its own, kept at a constant size, makes it easier to focus on the movement itself.

Here's a quick demonstartion of the app:

https://github.com/user-attachments/assets/469aa93b-4e6f-4010-9f4e-7ca49b612e33

The left panel is blured for privacy reasons. Normally, there is an original vieo overlayed with the skeleton that is shown on the right panel. It is used for observing whether the keypoints are placed right.

### How it helps in practice

The visualisation gives clinicians something concrete to point at while explaining. For example, a clinician could say: _"Look at the blue line, which is the left arm. You can see the long trail behind it, so it is moving a lot. The orange line, the right arm, barely moves."_ From there, the clinician can discuss what this might mean, for instance that the child may be relying more on one side of the body, using plain words that parents can follow.

It can also support clinicians themselves, especially those early in their careers. The tool isolates the keypoints that matter for gait and shows the walking pattern clearly, which makes it easier to notice and describe what is happening.

Understanding the gait pattern matters for everyone involved. Based on their assessment, the clinician can identify what the problem is and recommend suitable exercises. A clear visualisation helps parents see what that assessment is based on and why the exercises are recommended.

> This tool is a visual aid for communication. It does not diagnose anything. Interpreting the patterns and deciding on treatment remain the job of the clinician.

## Approach

### System overview

The tool is a web app. The interface is built with **Streamlit**, a Python framework for web frontends, so clinicians open it in a browser on a phone or computer and upload a video. **Tailscale**, a private encrypted network, lets phones reach that machine without exposing the app to the public internet, which matters because the data involves children.

### Computer vision

Computer vision is the part of this project I was most interested in. The quality of the visualisation depends entirely on how accurately the body keypoints are found, so I tested models to decide which is the best one.

#### Attempt 1: MediaPipe

I started with MediaPipe's **Pose Landmarker** (the Lite variant, through the MediaPipe Tasks API), which predicts 33 body landmarks per person and is designed to run in real time. I ran it in video mode on the test clips.

It was not accurate enough for this purpose. In some frames it placed the child's limbs inaccurately, so the skeleton did not follow the real walking pattern. Moreover, its foot representation was limited to two points per foot (heel and toe tip), which did not capture important foot movements during gait.

#### Attempt 2: RTMW

I then moved to **RTMW**, a whole-body pose model, run through the rtmlib library. It suited the project for several reasons:

- It predicts **133 keypoints**, including three points per foot (big toe, small toe and heel), which gives a more detailed description of foot movement than MediaPipe’s two.
- It uses a convolutional neural network (CNN) backbone to extract visual features, then predicts where each joint is. CNNs are good at recognising local patterns such as limbs, which makes the keypoints stable on a moving child.
- It is designed to be fast, which matters for processing videos in a clinical setting.

#### Comparing model settings

I compared two configurations on the same set of videos: **balanced on CPU** and **performance on GPU (CUDA)**.

- **Speed:** processing time was similar. The mean was about 51 s for balanced/CPU and 58 s for performance/GPU, so the GPU gave no end-to-end speedup. Most of the pipeline (video decoding, smoothing, rendering and encoding) runs on the CPU, so it likely limits the total time.
- **Accuracy:** on one clip, body keypoints differed by about 1.5% of torso length and foot keypoints by about 2.8%. Performance showed about 26% less frame-to-frame noise.

The visible differences were small, so I chose balanced on CPU. It's faster and easier to deploy.

## Reviews and future work

The completed app was demonstrated to clinicians and parents, who then tested it and provided feedback. Both sides said that this app helps a lot. It captures the main points of the body accurately and the visualisations are very clear. One client shared that the clinician's use of visualisations during consultations made the explanations easier to understand. The visual representation helped translate clinical observations into something more straightforward and accessible.
Another parent noticed that the visualisation helped her recognise similarities between her child's walking patterns and her husband's, particularly in their arm swing and body movement. This observation highlighted how simplified visual representations can make movement patterns easier to recognise. While certain gait characteristics may be quite apparent in adults, identifying them in children can be more challenging, as their smaller body proportions can make subtle movement patterns harder to distinguish.

Clinians and parents gave their review on this tool and suggested some ideas. Based on their advices, I plan to add next features in the future:

- **Step detection and cadence:** detect individual steps from the ankle and heel trajectories and calculate how many steps per minute the child takes.
- **Symmetry metrics:** compare left and right sides (step timing, limb movement, arm swing) and show the result as a simple number or chart next to the visualisation. This would back up what the trails already show visually.
- **Side-view joint angles:** with the child filmed from the side, calculate knee and hip flexion over the walking cycle. Joint angles from a single camera are only meaningful in the plane facing it, so these will need a standard side-view recording.
- **Saved sessions:** store results per patient and compare sessions over time, for example before and after a course of exercises.
