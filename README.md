# Billboard Detection Backend — Jabalpur (Hackathon MVP)

This repository contains the backend server for the Unauthorized Billboard Detection system, built for a hackathon. The server is responsible for processing uploaded images, detecting billboards, checking for regulatory compliance, and providing data to a reporting portal.

Day 1: infra + DB schema + storage + repo init.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   [Python 3.10+](https://www.python.org/downloads/)
*   [Pip](https://pip.pypa.io/en/stable/installation/)
*   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd billboard
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On macOS and Linux:**
```sh
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```sh
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The application requires several environment variables for configuration, including database credentials and API keys. These are managed using a `.env` file.

First, create a `.env` file in the root of the project:

```sh
touch .env
```

Now, open the `.env` file and add the following variables. You will need to get these values from your Supabase and Upstash project dashboards.

```
# Supabase Configuration
SUPABASE_URL="YOUR_SUPABASE_URL"
SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"

# Redis Configuration (for background tasks)
REDIS_URL="YOUR_UPSTASH_REDIS_URL"
```

**Important:** The `.env` file contains sensitive credentials and should **never** be committed to version control. The `.gitignore` file is already configured to ignore it.

### 5. Set Up the Database

The initial database schema is located in `infra/schema.sql`. You need to run this SQL script on your PostgreSQL database, which you can do via the Supabase dashboard.

1.  Log in to your [Supabase account](https://app.supabase.io/).
2.  Navigate to your project's **SQL Editor**.
3.  Copy the contents of `infra/schema.sql` and paste them into the editor.
4.  Click **"Run"** to create the necessary tables.

### 6. Running the Application

The main application is built with FastAPI. To run the development server:

```sh
uvicorn src.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`, and the auto-generated documentation can be found at `http://127.0.0.1:8000/docs`.

### 7. Running Tests

To run the test suite, use `pytest`:

```sh
pytest
```
