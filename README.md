# nanoGPT with Triton Kernels(WIP)

## Description
nanoGPT with Triton Kernels is a specialized implementation of the GPT (Generative Pre-trained Transformer) model optimized with Triton, a programming language and compiler designed to enhance computational performance on GPUs. This project aims to demonstrate significant improvements in model training and inference speed by leveraging Triton's capability to efficiently manage GPU resources.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Installation
To set up this project locally, follow these steps:

```bash
# Clone the repository
git clone git@github.com:your-username/nanoGPT-Triton.git
```
# Navigate to the project directory
```bash
cd nanoGPT-Triton
```
# Install required dependencies
```bash
pip install -r requirements.txt
```
## Usage
 To run the nanoGPT model with Triton kernels, execute:

 ```bash
 python run_model.py
```
## Features

•	Implementation of GPT with optimized attention and normalization layers using Triton.
•	Enhanced performance on GPU computations.
•	Easy integration with existing PyTorch models.

## License

Distributed under the MIT License. See LICENSE for more information.

## Credits
•	Triton programming language and tools (https://triton-lang.org)
•	OpenAI for the GPT model architecture.
•   Andrej Kapathy nanoGPT implemetation.
•   Tunadoable Triton tutotials.



