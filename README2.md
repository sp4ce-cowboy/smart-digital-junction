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
- apt list = \<link to file\>
- Technical specs
	- Boot instructions
	- 128GB SSD
	- Some useful software installed (screen cap stuff)
	- Passwords

#### Hailo-8
- hailortcli fw-control identify = \<INSERT SCREENSHOT\>
- 

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

**Object Detection**

The repository contains standard object detection example applications. These demonstrate how frames are captured, resized, and run through the Hailo pipeline.

This is also the file that is modified for the smart digital junction use case (which will be explained below)

**Instance Segmentation, Pose Segmentation, and Advanced Use Cases**

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

More information about the code is explained as inline documentation i.e. code comments.

### Overview of Approach
**The main concept is basically**

1.	Use the existing Hailo pipeline to get detections from the .HEF model instead of using ultralytics.YOLO.

2.	Convert those detections into supervision.Detections objects and process them similarly to how it’s done in the standalone code.

3.	Run ByteTrack for tracking and perform speed estimation on the tracked objects.

4.	Annotate the frames with bounding boxes, labels (including speed), and optionally traces.

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




