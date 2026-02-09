# FYSearch

CPU-only offline multimodal forensic search engine (OCR + embeddings + vector search).

---

## Supported Platforms

| Platform | Status |
|----------|--------|
| 🐧 Linux | ✅ Fully Supported |
| 🪟 Windows | ✅ Supported |
| 🍎 macOS | ✅ Supported |

---

## Prerequisites

### System Dependencies

Before installing, you need **Tesseract OCR** and **Poppler** installed on your system.

<details>
<summary><strong>🐧 Linux (Debian/Ubuntu)</strong></summary>

```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

**Option 1: Using Chocolatey (recommended)**
```powershell
choco install tesseract poppler
```

**Option 2: Manual Installation**
- Tesseract: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Poppler: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)

> ⚠️ Add both to your system PATH after installation.

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
brew install tesseract poppler
```

</details>

---

## Installation

### Step 1: Create Virtual Environment

<details>
<summary><strong>🐧 Linux / 🍎 macOS</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If you get an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

</details>

<details>
<summary><strong>🪟 Windows (Command Prompt)</strong></summary>

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

</details>

### Step 2: Install Dependencies

```bash
pip install -e ".[web,embeddings,faiss,ocr,pdf_images]"
```

---

## Running the Application

### Start the Web Server

<details>
<summary><strong>🐧 Linux / 🍎 macOS</strong></summary>

```bash
source .venv/bin/activate && fysearch web --host 127.0.0.1 --port 8000
```

</details>

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
.venv\Scripts\Activate.ps1; fysearch web --host 127.0.0.1 --port 8000
```

</details>

<details>
<summary><strong>🪟 Windows (Command Prompt)</strong></summary>

```cmd
.venv\Scripts\activate.bat && fysearch web --host 127.0.0.1 --port 8000
```

</details>

### Access the Application

Open in browser: **http://127.0.0.1:8000**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `tesseract not found` | Ensure Tesseract is installed and added to PATH |
| `poppler not found` | Ensure Poppler is installed and added to PATH |
| Virtual env won't activate (Windows) | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port already in use | Change `--port 8000` to another port (e.g., `--port 8080`) |

