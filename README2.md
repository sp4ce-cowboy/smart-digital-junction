# AI-on-Edge: Smart Digital Junction

## Documentation Overview
### Context
Running Object Detection Models for real-time applications with a sufficiently high accuracy is computationally expensive and consequently, fiscally expensive. The **Hailo-8L AI Processor** is a Neural Processing Unit that allows for such applications to be executed in a cost-efficient and energy-efficient manner, in comparison to conventional Desktop + discrete GPU environments.

This is made possible through a unique combination of hardware and software integration. Hailo uses a special Dataflow Compiler that can convert ODMs in the universal ONNX format to the proprietary Hailo Executable File (HEF) Format. This allows for faster execution of the same ODM while preserving detection accuracy, all the while 

This write-up covers the process of setting up the NPU to running a basic real-time application on the Hailo processor.

### Target Audience
This write-up is intended for any user, equipped with a Raspberry 5 Pi microcomputer and a Hailo-8 NPU, intending to develop and run custom ML applications, specifically for real-time use cases. 

More specifically, this write-up is written with the intention of conveying to a developer, proficient in software development (Python3.x.x OOP, Unix, Shell, and ChatGPT) and without any pre-existing ML knowledge, the initialization process of setting up the Hailo processor, and running some basic programs on it, to get the hang of what’s going on and being able to use this as a foundation for further exploratory developments. 

This write-up is NOT intended for any user who is intending to 
- Write custom python API wrappers over the base C++ code
- Run multiple models concurrently on the same Hailo-NPU
- Run any non-ONNX models on the Hailo processor
- Develop any otherwise specific and unorthodox applications

Nonetheless, this write-up will be useful for anyone intending to learn more about the basics of the Hailo-8 processor and running/developing basic applications for this platform (or any other purpose) written from a software engineering perspective.

_Additionally, this write-up is also intended as a handover to the next software engineer that will be working on this project after me, so it will contains some aspects that might not make sense to the general developer._

### Disclaimer
The information here is intended to be a progress report, of what worked and what didn’t work, what problems were solved and what weren’t, and how to set up a basic working environment, and how to make sense of this environment in order to develop it further. This documentation is a personalized depiction of the entire process and architecture. It is not intended to replace Hailo’s documentation entirely, and the information here can become outdated depending on the development of Hailo’s software. 

It is very important to pay attention to the software package versioning and dependencies to be able to replicate this. As the software provided by the company is in a developmental state, it is prone to changes and version tracking is largely experimental, programs and packages can fail to work with minor changes and it is not clear at this point to what extent these changes are reversible and to what extent these changes can permanently affect the hardware/firmware. This information therefore is provided at an AS-IS basis. Developer discretion is advised.

### Table of Contents

- [AI-on-Edge: Smart Digital Junction](#ai-on-edge-smart-digital-junction)
  - [Documentation Overview](#documentation-overview)
    - [Context](#context)
    - [Target Audience](#target-audience)
    - [Disclaimer](#disclaimer)
    - [Table of Contents](#table-of-contents)
  - [Smart Digital Junction Context](#smart-digital-junction-context)
    - [Set-up](#set-up)
      - [Setting up the Raspberry Pi and Hailo](#setting-up-the-raspberry-pi-and-hailo)
      - [Versions & Environment Specifications](#versions--environment-specifications)
        - [Raspberry Pi 5](#raspberry-pi-5)
        - [Hailo-8](#hailo-8)
        - [Verifying Proper Installation](#verifying-proper-installation)
  - [Hailo Programming Context](#hailo-programming-context)
    - [Running Basic Examples](#running-basic-examples)
    - [Cloning the rpi-5 Examples Repository](#cloning-the-rpi-5-examples-repository)
      - [Object Detection](#object-detection)
      - [Instance Segmentation, Pose Segmentation, and Advanced Use Cases](#instance-segmentation-pose-segmentation-and-advanced-use-cases)
  - [Speed Estimation](#speed-estimation)
    - [Setting up the Application](#setting-up-the-application)
    - [Running the Application](#running-the-application)
    - [Overview of Approach](#overview-of-approach)
    - [Practical Explanation of Code](#practical-explanation-of-code)
    - [Mathematical Explanation](#mathematical-explanation)
  - [Speed Estimation Benchmarking](#speed-estimation-benchmarking)
    - [Power Consumption Analysis](#power-consumption-analysis)
    - [Resources Consumption Analysis](#resources-consumption-analysis)
- [Dataflow Compiler Context](#dataflow-compiler-context)
  - [Setting up Dev Environment](#setting-up-dev-environment)
    - [Custom Python Binary](#custom-python-binary)
    - [Entering the Dev Environment](#entering-the-dev-environment)
  - [Dataflow Compiler Context](#dataflow-compiler-context-1)
    - [Manual Pipeline](#manual-pipeline)
    - [Hailo Model Zoo Pipeline](#hailo-model-zoo-pipeline)
  - [Theoretical Limitations](#theoretical-limitations)
- [Conclusion & Next Steps](#conclusion--next-steps)
  - [Updating Firmware & Running Examples](#updating-firmware--running-examples)
  - [Updating the Speed Estimation Program](#updating-the-speed-estimation-program)
  - [Advanced Examples](#advanced-examples)
  - [Advanced Model](#advanced-model)

## Smart Digital Junction Context
The aim of this project is to perform real-time monitoring of traffic, to retrieve useful data about traffic flow that can be used as a foundation for further analytic. Object Detection Models with labels restricted to road traffic context can be applied for video analytics frame-by-frame, but real-time monitoring requires a minimum of 25 FPS to be effective. To run an ODM to attain this performance level would require an amortized inference rate of 0.07 seconds per frame. This is possible with a standard Desktop + GPU + CPU Environment, however, incurring large amounts of capital cost and operation costs in the process. 

Hailo offers a work around by providing specialized hardware and the tools to create specialized software that can run on their hardware. Doing so would drastically minimize the operation costs by reducing the power consumption to a fraction of the initial amount, with negligible loss in inference accuracy and validity. This project is an exploration of the suitability and applicability of Hailo processors to achieve the above mentioned results.

Additionally, Hailo’s specialized software also allows developers to create applications for their own nuanced use cases as opposed to relying on standardized configurations provided by off-the-shelf Video Analytics (VA) providers.

In this documentation is articulated the process of creating an application that is capable of performing speed estimation using Hailo’s hardware (i.e. Hailo-8 NPU) via Hailo’s software (i.e. Hailo Model Zoo, Hailo Dataflow Compiler, Hailo RT CLI).

## Set-up
### Setting up the Raspberry Pi and Hailo
The instructions for setting up the Raspberry Pi 5 and Hailo can be found on Hailo’s GitHub Repo HERE \<INSERT LINK\>. Take note of the environment specifications below before making any permanent installations. 

### Versions & Environment Specifications
The versions requirement might have changed to the latest. The later versions may or may not support the application developed with the versioning of the current working environment. These are:

#### Raspberry Pi 5
- uname -a = \<INSERT SCREENSHOT\>
- neofetch output = \<INSERT SCREENSHOT\>
![IMG_8512](https://github.com/user-attachments/assets/22f660f6-c277-406d-bc98-39f3081ce360)
- apt list = \<link to file\>
- Technical specs
	- Boot instructions
	- 128GB SSD
	- Some useful software installed (screen cap stuff)
	- Passwords

#### Hailo-8
- hailortcli fw-control identify = \<INSERT SCREENSHOT\>
  

#### Verifying Proper Installation
Before proceeding, ensure your Raspberry Pi has Raspberry Pi OS installed, with minimally 128 GB and an internet connection. Check if the Hailo chip is properly connected by running this command:

```swift
hailortcli fw-control identify
```

The output should be identical to the above environment specifications, and any errors must be debugged before you proceed, which can be found here.

INSERT LINK TO HAILORT DEBUG

## Hailo Programming Context
#### Running Basic Examples

Once your Pi recognizes the Hailo device, begin by running the sample applications included in the Hailo rpi examples repo. These demonstrate simple object detection tasks.

You can clone the repo at this link \<INSERT LINK\> and follow the instructions to execute the basic object detection model. All the HEF models and video files will be included in the installation.

Remember to enter the python virtual environment FIRST!

#### Cloning the rpi-5 Examples Repository

Hailo hosts a dedicated set of Raspberry Pi 5 examples. To avoid version mismatches, clone the repository’s main branch and checkout to the relevant older commit.

For the current set up, the commit is dated at 8 AUGUST 2024 
(VERIFY THIS)

If you need older or newer versions, use Git’s checkout command to revert to previous commits.

#### Object Detection

The repository contains standard object detection example applications. These demonstrate how frames are captured, resized, and run through the Hailo pipeline.

This is also the file that is modified for the smart digital junction use case (which will be explained below)

#### Instance Segmentation, Pose Segmentation, and Advanced Use Cases

Hailo also publishes more advanced examples like instance segmentation for detecting object masks and pose estimation for tracking keypoints of humans. Investigate these for broader applications, including posture analysis or crowd counting.

## Speed Estimation
Refer to the code provided in this repo. \<INSERT LINK\>.

This code is a combination of both the Roboflow’s speed estimation code (INSERT LINK) and Hailo’s default object detection model process.

\<ELABORATE MORE BELOW\>

### Setting up the application
1. Clone this repo
2. `cd` into the hailo-rpi5-examples folder
3. Enter the python virtual environment with `source setup_env.sh`_ 1. REMEMBER THAT SH IS NOT FORWARDS COMPATIBLE WITH ZSH!
	2. You need to explicitly enter bash if your default shell is zsh
	3. Only execute the above command after you have entered `sh`
	4. If this command does not output a Hailo version, then you need to return to the Hailo intialization process and redo it.
4. run `pip install -r requirements_with_supervision.txt`

### Running the Application
1. `cd` into the hailo-rpi5-examples folder
2. Enter the virtual environment
3. Run this command

```bash
python basic_pipelines/detection_supervision_tappas328.py --input resources/video1.mp4 --hef-path resources/yolov8m.hef --use-frame
```

<img width="1060" alt="image" src="https://github.com/user-attachments/assets/0f8b4f18-2d02-49ff-9aff-b8cb0f4aa605" />


More information about the code is explained as inline documentation i.e. code comments.

### Overview of Approach
**The main concept is basically**
1. Use the existing Hailo pipeline to get detections from the .HEF model instead of using ultralytics.YOLO.
2. Convert those detections into supervision.Detections objects and process them similarly to how it’s done in the standalone code.
3. Run ByteTrack for tracking and perform speed estimation on the tracked objects.
4. Annotate the frames with bounding boxes, labels (including speed), and optionally traces.

**Key Points:**
- The original supervision code relied on YOLO("yolov8x.pt") to get detections. Here I only rely on the Hailo pipeline’s output (already parsed as detections = roi.get\_objects\_typed(hailo.HAILO\_DETECTION)).

- Integration of ViewTransformer, PolygonZone, ByteTrack, and annotators from the supervision code into the hailo detection pipeline.

- The generation of the .HEF file is explained later, but for now set --hef-path to load it directly as the compiled models are available in the resources folder. Once loaded, the pipeline produces detections that can be used directly.

- We store and initialize objects like ByteTrack, PolygonZone, etc (and smth else add later), in GStreamerDetectionApp. Then the logic of speed estimation is integrated inside the callback (app\_callback) where there is access to frame data and detections.

- MAKE SURE supervision, ultralytics, and related dependencies are installed in your environment. All the dependencies are VERSION CORRECTED inside the requirements_with_supervision file. Although here ultralytics is not needed since we’re not using their model, supervision and ByteTrack from supervision are necessary.

- Adjust parameters (FPS, polygon coordinates, thresholds) as needed. How to do this is explain as inline documentation.

### Practical Explanation of Code
**What this \<INSERT LINK\> code does:**
1. Uses the Hailo pipeline to run a .HEF model and get detections.
2. Converts these detections into supervision.Detections.
3. Applies the same logic as the original supervision code: polygon filtering, NMS, ByteTrack tracking, speed estimation via coordinates history, and annotations (boxes, labels, speeds).
4. Displays the annotated frames via the Hailo pipeline’s sink or the User Frame window (if enabled with --use-frame).

**Possible Modifications:**
- Adjust the SOURCE polygon coordinates, TARGET dimensions, frame rate (user\_data.fps), and any other parameters to match the application.
- The .HEF model provides the option to use custom class labels which can be integrated into the logic for class\_ids. This will be explained later.
- Ensure the input format (NV12/YUYV, which are some standard machine vision formats that you can read up about separately) is converted correctly to BGR if needed. In the example above, the input is RGB, but this might not always be the case.

### Mathematical explanation
The initial speed estimation algorithm is taken from Roboflow’s article [here](https://blog.roboflow.com/estimate-speed-computer-vision/).

Essentially, this code converts a given video file containing a head-on view of a road (which would appear as a polygon) into a "bird's eye" representation of the road. This requires information about the location of the video, such as the width of the road and the actual distance of the road in view of the camera. This information is then linearly mapped and scaled to the pixel count, afterwhich basic arithmetic can be used to calculate the speed of an object (i.e. distance/time or inverse of frame rate).

```python
# ---------------------- ADDITIONAL CODE FOR SPEED ESTIMATION ------------------------------------

# Define the polygon in source perspective
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
SOURCE = SOURCE.astype(float)
TARGET_WIDTH = 22 # WIDTH OF THE ROAD IN METERS
TARGET_HEIGHT = 80 # DISTANCE OF THE ROAD IN METERS
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)
```

#### IMPORTANT!
- The SOURCE vertices must be manually determined at the point of calibration.
- As such, this application is unsuitable for usecases where the camera or input source is non-stationary, or used in varying terrains.
- The Hailo pipeline compresses all videos into a 640x640 resolution, while preserving the original aspect ratio of the input source.
- This means that the video will have 2 black bars above and below the actual video that is playing, which must be taken into account during the scaling.

## Speed Estimation Benchmarking
### Power Consumption analysis
The power consumed by the Hailo Processor can be retrieved with: 
```shell
hailortcli measure-power
```

### Resources Consumption Analysis
The resources consumed by the Hailo Processor can be retrieved with:
```shell
hailortcli monitor
```

Note that for both use cases, the actual application must be running concurrently in a different terminal window.
---
# Dataflow Compiler Context
- Dataflow Compiler Context
	- Setting up Dev environment
		- Kali works fine but might not
	- installation of Dfc (3.27) and link to compatible versions on hailo forum
	- Compiling a new version of python for dfc
	- Installation of hailo model zoo
	- Making sense of the dataflow compiler and model zoo
		- Include archi diagram
	- Link to custom scripts for model zoo vs DFC (the edge impulse example)
	- Compiling a basic yolov6m model
		- Energy restrictions, max is yolov8l
	- Testing out trainin DFC model and verifying that basic examples work

# Dataflow Compiler
The dataflow compiler (DFC) is the desktop counterpart to Hailo's realtime environment. The DFC is used to convert open source Open Neural Network Exchange form at (ONNX) models into Hailo Executable File format (HEF). A sequence of processes - optimization, quantization and compiling - allow a heavy ONNX model to run on a light RT environment like the Hailo.

## Setting up Dev Environment (from scratch)
Setting up the development environment is relatively straightforward compared to setting up the RT environment.

1. Check the correct versions required from \<THIS LINK to the chart\>
	1. For the current Raspberry Pi, this is DFC v 3.27.0
2. Install the appropriate versions from this link
3. Before following the installation instructions, make sure to have the correct python binary installed. With the appropriate version of python installed, you need to create a virtual environment using the following command:
	```bash
	python3.x -m venv NAME_OF_ENV
	# replace x above with 7, 8, 9, 10 or 11 accordingly
	# For DFC 3.27, this is python 3.8
	```
4. Enter the virtual environment and proceed with the installation of the dataflow compiler following the instructions from the Repo.

### Custom Python Binary
You might have to install some tools to be able to compile your own python version, there are instructions available for this but GPT-4o/o1 is reliable enough to take you through the step by step process.

### Entering the Dev Environment
The above instructions are for setting up the environment from scratch. However, the Kali workstation comes with DFC preinstalled. To use it, follow these steps:

1. `cd` into the dfc_3_27 folder on the Desktop
2. Enter shell with `sh`
3. Run the virtual environment source command `source bin/activate` 

The DFC environment should now be available to use. 

## Dataflow Compiler Context
There are two main pipelines through with ONNX models get converted to HEF models. The first is the Hailo Model Zoo (HMZ) pipeline and the second is the manual pipeline.

The below diagram shows the steps involved behind the scenes in the conversion process.

\<INSERT DIAGRAM\>

### Manual Pipeline
As shown in the above diagram, the manual pipeline involves manually going through each process using a dedicated script for each process - from optimization to quantization to compilation. 

The scripts are included in the \<INSERT LINK TO FOLDER\>, however, take note that these are draft scripts modeled after the EGDE IMPULSE project \<INSERT LINK\>. While the scripts run successfully, the compilation is generally unsuccessful, and the manual pipeline was not used in achieving the final compiled HEF format that we used to benchmark the Hailo Processor.

However, feel free to modify the scripts such that a favourable compilation result is produced.

### Hailo Model Zoo Pipeline
The HMZ pipeline is basically an abstraction over the manual pipeline. Instead of running each step through a script, each script is minimized into a simple command line command. 

For example:
- `hailomz optimize`
- `hailomz compile`
- `hailomz profile`

These commands allow for the functionality of the original manual pipeline to be compressed into easily executable commands. However, the limitation is that only the hailomz models listed through the `hailomz list` command are available for compilation, and these models cannot be modified in any way.

In order to compile custom models or models outside the supported models list, the manual pipeline must be used.

In the Kali workstation, after entering the DFC virtual environment, the above `hailomz` commands can be executed. The below command compiles a YOLOv8s model:

```bash
hailomz compile yolov8s
```

You can find information about each model using the `hailomz info` command. **Take note that the Hailo-8 processor can only support up to YOLOv8m for dense traffic analysis before overheating and terminating the network. **

The compilation process can take up to 24 hours in some cases. The energy settings for the Kali workstation have been modified such that the standby process does not interrupt compilation, but to be sure that the system settings do not override this, run `caffeine` in another terminal.

The output of the compilation process is a ready-to-use HEF file, which can be transferred to the resources folder of the speed estimation code, and invoked with the —hef-path flag as mentioned above.

## Theoretical Limitations
The below diagram explains the ML-specifics of the Dataflow compiler. 

\<Insert Guan Jie’s diagram\>

# Conclusion & Next Steps
This demarcates the end of this documentation. Through these processes elaborated above, you should be familiar with:

1. Compiling a simple HEF model using the Hailo Model Zoo.
2. Running the speed estimation program with the compiled model
3. Running some power/resources consumption measurements  

This project has a few natural pathways for extension going forward.

#### Updating firmware & running examples
The current HailoRT 4.17 does not fully support the Hailo API. The first step would be to update all components (i.e. HailoRT CLI, DFC, Hailomz, etc) to run with the latest software versions. 

This would entail re-cloning the latest version of the `hailo-rpi5-examples` repository that includes support for the later software and firmware versions.

#### Updating the Speed Estimation program
By this point, the existing speed estimation program will fail to work. It is therefore necessary to 1) Find the updated `hailo-rpi-examples` code, specifically `detection.py` and 2) combine this with the Roboflow supervision code shown earlier. 

This can be done manually, however, given the limitations of Hailo’s API documentation, it would be better to prompt an LLM with the project context, the detection.py code and the supervision code, and ask the LLM to  merge the two together and afterwards manually sort out any errors that might arise.

#### Advanced Examples
You can try to run more advanced programs, like the License Plate Recognition (LPR) and Facial Detection examples, and modify them to fit your specific use case.

#### Advanced Model
You can try to follow the DFC manual pipeline for any model of your choice, such that you end up with a HEF that you can use with any of the examples above.

---
# drafts below

### Table of Contents
- Smart Digital Junction Context X
- Setting up the Raspberry Pi + Hailo X
	- Pre-existing raspberry Pi set up and link to hailo instructions X
	- Checking if hailo is properly connected X
	- Version conditions  X
	- All other prerequisites for proceeding.  X
- Hailo Programming Context
	- Running basic examples X
	- Cloning the rpi-5 examples repo X
		- Disclaimer about versioning and etc X
		- Cloning the correct repo X
		- Using Git to find older versions of the repo X
	- Object Detection X
	- Instance segmentation X
	- Pose segmentation X
	- Finding examples of other advanced use cases (cancel)
- Custom Development
	- Smart Digital Junction & Speed estimation use case X
	- Roboflow example & Math behind speed estimation (LATER)
	- Software engineering explanation X
	- Include some basic diagrams.  (LATER)
	- Explanation of Combined roboflow + hailo basic examples and the command line invocation X
- DOCUMENTATION/REFERENCE OF ACTUAL CODE 
	- Upload the whole code and add code comments (and also upload the resources folder) X
	- Explain each part like coloring, etc) X
	- Process of running (like entering the venv, which python version, sourcing, installation etc) X
	- command line invocation (and explanation) X
	- Disclaimer about old code used and newer code not used) X

- Dataflow Compiler Context
	- Setting up Dev environment
		- Kali works fine but might not
	- installation of Dfc (3.27) and link to compatible versions on hailo forum
	- Compiling a new version of python for dfc
	- Installation of hailo model zoo
	- Making sense of the dataflow compiler and model zoo
		- Include archi diagram
	- Link to custom scripts for model zoo vs DFC (the edge impulse example)
	- Compiling a basic yolov6m model
		- Energy restrictions, max is yolov8l
	- Testing out trainin DFC model and verifying that basic examples work


- Analysing the Run
	- Power analysis
	- Resources/CPU monitoring
	- Hailortcli fw-control —help (everything)
- Conclusion
	- Energy saving (and power point slides)
	- Further applications
	- Miscellaneous notes
		- Temp space expansion
		- 128gb rasp pi
		- Initial installation unknown but best to install full
		- Hailo wrapper over c++
	- Useful links
		- Versioning board link
		- Downloading from archived on HDZ


