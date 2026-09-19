# IA Claude

IA Claude is a terminal-based coding agent for exploring and working with a
codebase.

## Project setup on Windows

### Requirements

Install the following tools:

- Git 2.40 or newer
- Python 3.12
- Node.js 20 or newer
- pipx

Verify Git, Python, and Node.js:

```powershell
git --version
py -3.12 --version
node --version
npx --version
```

### 1. Install pipx

```powershell
py -3.12 -m pip install --user pipx
py -3.12 -m pipx ensurepath
```

Close PowerShell and open it again, then verify the installation:

```powershell
pipx --version
```

### 2. Clone the project

```powershell
git clone https://github.com/ianuj4231/coding_agent_cli.git
cd coding_agent_cli
```

### 3. Install IA Claude

Install the project in an isolated Python environment:

```powershell
pipx install --python (py -3.12 -c "import sys; print(sys.executable)") --editable .
```

### 4. Verify the installation

```powershell
pipx list
Get-Command ia_claude
```

The output should list `claude-rag-agent` and the `ia_claude` command.

### 5. Create the environment file

Copy the example file to `.env` in the project root:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder values with your own credentials:

```powershell
notepad .env
```

Never commit `.env` or share its credentials.

## Project setup on macOS

### 1. Install the required tools

Install Homebrew from [brew.sh](https://brew.sh/) if it is not already
available, then run:

```bash
brew install git python@3.12 node pipx
pipx ensurepath
```

Close Terminal and open it again, then verify the installations:

```bash
git --version
python3.12 --version
node --version
npx --version
pipx --version
```

### 2. Clone the project

```bash
git clone https://github.com/ianuj4231/coding_agent_cli.git
cd coding_agent_cli
```

### 3. Install IA Claude

```bash
pipx install --python "$(command -v python3.12)" --editable .
```

### 4. Verify the installation

```bash
pipx list
command -v ia_claude
```

The output should list `claude-rag-agent` and the `ia_claude` command.

### 5. Create the environment file

Copy the example file to `.env` in the project root, then edit it:

```bash
cp .env.example .env
nano .env
```

Replace the placeholder values with your own credentials. Never commit `.env`
or share its credentials.
