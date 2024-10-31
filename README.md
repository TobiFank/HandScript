# HandScript

![License: Custom Non-Commercial](https://img.shields.io/badge/License-Custom%20Non--Commercial-red.svg)

HandScript is an intelligent handwriting recognition system that transforms handwritten documents into editable digital text, learning and adapting to your unique handwriting style.

![HandScript Dashboard](./docs/images/dashboard-preview.png)

## Features

### 🔍 Smart Document Organization
- Organize documents into projects for easy management
- Drag-and-drop multiple page upload
- Intuitive page reordering
- Export to PDF or Word formats
- Built-in document preview

![Document Management](./docs/images/document-management.png)

### 🤖 Personalized Recognition
- Individual writer profiles with custom recognition models
- Continuous learning from your handwriting
- Support for English and German text
- Performance tracking and analytics
- Visual recognition insights

![Writer Profiles](./docs/images/writer-profiles.png)

### ✏️ Interactive Training
- Simple training process with your handwriting samples
- Real-time feedback on recognition quality
- Visual line detection and segmentation
- Easy verification workflow
- Progress tracking and model improvement

![Training Interface](./docs/images/training-interface.png)

### 📋 Page Management
- Drag-and-drop reordering of pages
- Visual thumbnail previews
- Quick page status overview
- Batch processing options
- Writer assignment per page

![Page Management](./docs/images/page-management.png)

### 📝 Powerful Editing
- Side-by-side comparison view
- Line-by-line verification
- Rich text formatting options
- Batch processing capabilities
- Visual text line highlighting

![Editor Interface](./docs/images/editor-interface.png)

### 👀 Document Preview
- Clean, formatted text preview
- One-click export to PDF or DOCX
- Page navigation
- Print-ready formatting
- Document statistics overview

![Document Preview](./docs/images/document-preview.png)

## System Requirements

### Recommended Hardware
- NVIDIA GPU with 8GB+ VRAM
- 16GB RAM
- 15GB free storage space

### Software
- Docker and Docker Compose
- NVIDIA drivers for GPU support
- Modern web browser (Chrome/Firefox/Safari recommended)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/TobiFank/HandScript.git
```

2. Create environment file:
```bash
cat > .env << EOL
DEVICE=cuda
DEFAULT_MODEL_ENGLISH=microsoft/trocr-large-handwritten
DEFAULT_MODEL_GERMAN=fhswf/TrOCR_german_handwritten
EOL
```

3. Start the application:
First time setup:
```bash
docker-compose up --build
```

Subsequent runs:
```bash
docker-compose up
```

4. Open your browser and navigate to:
```
http://localhost:3000
```

## Usage Guide

### Setting Up Projects
1. Click "New Project" on the dashboard
2. Enter project name and optional description
3. Upload documents to your project
4. Organize pages within documents

### Creating Writer Profiles
1. Navigate to Writers section
2. Click "Add Writer" to create a new profile
3. Select language (English/German)
4. Add training samples to improve recognition

### Training Process
1. Upload handwriting samples
2. Verify recognized text
3. Monitor training progress
4. Track accuracy improvements
5. Export processed documents

### Document Processing
1. Upload document pages
2. Assign writer profile
3. Process pages automatically
4. Review and edit recognized text
5. Export final document

## Perfect For
- ✨ Researchers digitizing handwritten notes
- 📚 Teachers processing student submissions
- 📑 Professionals managing handwritten documents
- 🏛️ Archivists converting historical documents
- 📝 Anyone needing to digitize handwritten text

## Directory Structure

```
handscript/
├── backend/          # FastAPI backend application
├── frontend/         # React frontend application
├── docker/          # Docker configuration files
├── storage/         # Data storage directory
│   ├── images/      # Uploaded document images
│   ├── models/      # Trained writer models
│   └── exports/     # Exported documents
└── docs/            # Documentation and images
```

## Architecture

![Handscript Architecture](./docs/images/architecture.png)

## Development
### Backend Development
```bash
cd backend
poetry install
poetry run python run.py
```

### Frontend Development
```bash
cd frontend
yarn install
yarn dev
```

## License
This project is licensed under a Custom Non-Commercial, Contribution-Based License.

### Key Points:
- **Private, non-commercial use** of this tool is permitted.
- **Modifications or enhancements** must be contributed to this project (e.g., through pull requests) to be approved by the project maintainer.
- **Commercial use** and creating derivative works for redistribution outside of this project are prohibited.
- **Contact for Commercial Use**: Companies or individuals interested in commercial use should contact Tobias Fankhauser at [Your Contact Information] for case-by-case consideration.

For full details, please refer to the [LICENSE](LICENSE) file.

### The following components are licensed under different terms:
- [fhswf/TrOCR_german_handwritten](https://huggingface.co/fhswf/TrOCR_german_handwritten) is licensed under the Academic Free License 3.0. See [link](https://huggingface.co/fhswf/TrOCR_german_handwritten/tree/main).
- [microsoft/trocr-large-handwritten](https://huggingface.co/microsoft/trocr-large-handwritten) is licensed under the MIT License. See [link](https://github.com/microsoft/unilm/blob/master/LICENSE).

For more details, refer to the included license files and documentation.

## Acknowledgments
- Microsoft's TrOCR model for base recognition capabilities
- FHSWF's TrOCR German model for additional language support
- Hugging Face Transformers library
- FastAPI and React for the application framework
- The open source community for various supporting libraries

## Contributing
We welcome contributions that enhance the tool! Please submit a pull request for any proposed changes or additions. All contributions must comply with the Custom Non-Commercial, Contribution-Based License outlined in the [LICENSE](LICENSE) file.

### Contributor License Agreement (CLA)
By contributing, you agree to the terms outlined in the [CLA](CLA.md). This agreement ensures that all contributions can be used in any future version of the project, including potential commercial versions. Please read the CLA before submitting your pull request.
