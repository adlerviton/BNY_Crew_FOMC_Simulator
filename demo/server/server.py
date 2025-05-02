# server/server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
import sys
import time
import threading
import re
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.bny_capstone_crew import BnyCapstoneCrew

# from simulation import SimulationManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://44.200.155.84"],  # frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # If sending fails, we'll handle the cleanup in the main WebSocket route
                pass


websocket_manager = WebSocketManager()


class SimulationManager:
    def __init__(self, websocket_manager):
        self.websocket_manager = websocket_manager
        self.selected_date = None
        self.crew_instance = None
        self.is_running = False
        self.current_task = None
        self.tasks_completed = []
        self.crew_results = {}
        self.error = None
        self.progress = 0

        self.task_sequence = [
            "probabilities_comment",
            "get_economic_suggestions",
            "regional_analysis",
            "academic_analysis",
            "central_analysis",
            "regional_discussion",
            "academic_discussion",
            "central_discussion",
            "regional_vote",
            "academic_vote",
            "central_vote",
            "other_summary",
            "vote_summary",
            "prediction_summary",
            "summary_final",
        ]

        # Create a mapping from task description phrases to task names for more robust identification
        self.task_description_map = {
            "Make a comment about the implied probabilities": "probabilities_comment",
            "Fed's chief economist, analyze the current economic situation": "get_economic_suggestions",
            "Regional Pragmatists, analyze the economist's proposed monetary policy solutions": "regional_analysis",
            "Academic Balancers, analyze the economist's proposed monetary policy solutions": "academic_analysis",
            "Central Policymakers, analyze the economist's proposed monetary policy solutions": "central_analysis",
            "Regional Pragmatists, respond to the initial analyses": "regional_discussion",
            "Academic Balancers, respond to the initial analyses": "academic_discussion",
            "Central Policymakers, respond to the initial analyses": "central_discussion",
            "Regional Pragmatists, it is time to cast your final vote": "regional_vote",
            "Academic Balancers, it is time to cast your final vote": "academic_vote",
            "Central Policymakers, it is time to cast your final vote": "central_vote",
            "draft a **formal public statement**": "other_summary",
            "prepare a JSON summary of the final votes": "vote_summary",
            "prepare a JSON summary of the end of 2025 rate predictions": "prediction_summary",
            "Combine the outputs of other_summary, vote_summary, and prediction_summary": "summary_final",
        }

        # Create a mapping from agent names/roles to their assigned tasks
        self.agent_task_map = {
            "analyst": [
                "probabilities_comment",
                "other_summary",
                "vote_summary",
                "prediction_summary",
                "summary_final",
            ],
            "economist": ["get_economic_suggestions"],
            "Regional_Pragmatists": [
                "regional_analysis",
                "regional_discussion",
                "regional_vote",
            ],
            "Academic_Balancers": [
                "academic_analysis",
                "academic_discussion",
                "academic_vote",
            ],
            "Central_Policymakers": [
                "central_analysis",
                "central_discussion",
                "central_vote",
            ],
        }

        # Current task by agent - helps track what each agent is working on
        self.current_agent_tasks = {
            "analyst": None,
            "economist": None,
            "Regional_Pragmatists": None,
            "Academic_Balancers": None,
            "Central_Policymakers": None,
        }

        # Initialize votes and predictions data
        self.votes_data = [
            {"member": "Regional Pragmatist", "vote": "-9999.00%", "policy_vote": ""},
            {"member": "Academic Balancer", "vote": "-9999.00%", "policy_vote": ""},
            {"member": "Central Policymaker", "vote": "-9999.00%", "policy_vote": ""},
        ]

        self.predictions_data = [
            {"member": "Regional Pragmatist", "prediction": "0.00%"},
            {"member": "Academic Balancer", "prediction": "0.00%"},
            {"member": "Central Policymaker", "prediction": "0.00%"},
        ]

        self.fomc_statement = ""

        # Conversation logs
        self.conversation_logs = []

        # Debug log
        self.debug_log = []

    def reset(self):
        """Reset the simulation state"""
        self.is_running = False
        self.current_task = None
        self.tasks_completed = []
        self.crew_results = {}
        self.error = None
        self.progress = 0

        # Reset current agent tasks
        self.current_agent_tasks = {
            "analyst": None,
            "economist": None,
            "Regional_Pragmatists": None,
            "Academic_Balancers": None,
            "Central_Policymakers": None,
        }

        # Reset votes and predictions
        self.votes_data = [
            {"member": "Regional Pragmatist", "vote": "-9999.00%", "policy_vote": ""},
            {"member": "Academic Balancer", "vote": "-9999.00%", "policy_vote": ""},
            {"member": "Central Policymaker", "vote": "-9999.00%", "policy_vote": ""},
        ]

        self.predictions_data = [
            {"member": "Regional Pragmatist", "prediction": "0.00%"},
            {"member": "Academic Balancer", "prediction": "0.00%"},
            {"member": "Central Policymaker", "prediction": "0.00%"},
        ]

        self.fomc_statement = ""
        self.conversation_logs = []
        self.debug_log = []

    def get_current_state(self):
        """Get the current state of the simulation for sending to clients"""
        return {
            "status": "running" if self.is_running else "idle",
            "currentTask": self.current_task,
            "tasksCompleted": self.tasks_completed,
            "progress": self.progress,
            "votes": self.votes_data,
            "predictions": self.predictions_data,
            "fomcStatement": self.fomc_statement,
            "error": self.error,
            "conversationLogs": self.conversation_logs,
            "debugLog": (
                self.debug_log[-20:] if self.debug_log else []
            ),  # Last 20 debug messages
            "agentTasks": self.current_agent_tasks,
            "selectedDate": self.selected_date,  # Add selected date to state
        }

    async def run_simulation(self):
        """Run the FOMC simulation"""
        if self.is_running:
            return

        if not self.selected_date or not self.crew_instance:
            self.error = "Please select a date before starting the simulation"
            await self.broadcast_state()
            return

        self.is_running = True
        self.error = None
        await self.broadcast_state()

        try:
            crew = self.crew_instance.crew()

            # first_task = self.task_sequence[0]
            # self.current_task = first_task
            # first_agent = self._get_agent_name_for_task(first_task)

            # if first_agent in self.current_agent_tasks:
            #     self.current_agent_tasks[first_agent] = first_task

            # self._add_to_conversation_logs(
            #     {
            #         "type": "task_start",
            #         "task": first_task,
            #         "agent": first_agent,
            #         "timestamp": time.time(),
            #         "description": self._get_task_description(first_task),
            #     }
            # )

            await self.broadcast_state()

            def run_crew():
                try:
                    crew = self.crew_instance.crew()
                    crew.task_callback = self.task_callback
                    crew.step_callback = self.step_callback
                    result = crew.kickoff()

                    # Ensure result is a properly formatted JSON string
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith("'") and result.endswith("'"):
                            result = result[1:-1]  # Remove surrounding quotes
                        if result.startswith('"') and result.endswith('"'):
                            result = result[1:-1]  # Remove surrounding quotes

                    self.crew_results = result
                    self.is_running = False
                    self.progress = 100

                    # Process the final results
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._process_final_results())
                        loop.run_until_complete(self.broadcast_state())
                    finally:
                        loop.close()

                except Exception as e:
                    self.error = str(e)
                    self.is_running = False

                    # Broadcast error state
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.broadcast_state())
                    finally:
                        loop.close()

            thread = threading.Thread(target=run_crew)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.error = str(e)
            self.is_running = False
            await self.broadcast_state()

    def task_callback(self, task, output=None, is_start=False):
        """Callback function for task execution with state tracking"""
        # Extract task name
        task_name = self._extract_task_name(task)
        agent_name = self._get_agent_name_for_task(task_name)

        if is_start:
            # Log task start
            self.current_task = task_name
            if agent_name in self.current_agent_tasks:
                self.current_agent_tasks[agent_name] = task_name

            log_entry = {
                "type": "task_start",
                "task": task_name,
                "agent": agent_name,
                "timestamp": time.time(),
                "description": self._get_task_description(task_name),
            }
            self._add_to_conversation_logs(log_entry)
        else:
            print(f"\n=== Task Callback for {task_name} ===")
            # Process task output for vote-related tasks
            if task_name in [
                "regional_vote",
                "academic_vote",
                "central_vote",
                "other_summary",
            ]:
                print(f"Processing output for {task_name}")
                self._process_task_output(task_name, output)

            # Update progress
            if task_name in self.task_sequence:
                task_index = self.task_sequence.index(task_name)
                self.progress = int((task_index + 1) / len(self.task_sequence) * 100)

            # Add to completed tasks if not already there
            if (
                task_name
                and task_name != "unknown_task"
                and task_name not in self.tasks_completed
            ):
                self.tasks_completed.append(task_name)

            # Add task completion to logs
            if output:
                log_entry = {
                    "type": "task_complete",
                    "task": task_name,
                    "timestamp": time.time(),
                    "content": str(output),
                }
                self._add_to_conversation_logs(log_entry)

        # Broadcast state update
        threading.Thread(target=self._broadcast_state_sync).start()
        return output

    def step_callback(
        self,
        agent=None,
        task=None,
        step=None,
        input_message=None,
        llm_response=None,
        output_message=None,
    ):
        """Callback function for agent steps"""
        # Get agent name
        agent_name = getattr(agent, "name", None) or getattr(
            agent, "role", "Unknown Agent"
        )
        task_name = self._extract_task_name(task)

        # Update current task for the agent
        if task_name != "unknown_task" and agent_name in self.current_agent_tasks:
            self.current_agent_tasks[agent_name] = task_name
            self.current_task = task_name

        # Add thinking step to logs if there's content
        if llm_response:
            log_entry = {
                "type": "agent_thinking",
                "agent": agent_name,
                "task": task_name,
                "timestamp": time.time(),
                "content": str(llm_response),
            }
            self._add_to_conversation_logs(log_entry)

        # Broadcast state update
        threading.Thread(target=self._broadcast_state_sync).start()
        return output_message if output_message is not None else llm_response

    def _broadcast_state_sync(self):
        """Helper to broadcast state in a synchronous context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.broadcast_state())
        finally:
            loop.close()

    def _extract_task_name(self, task):
        """Extract task name from task object"""
        if not task:
            return "unknown_task"

        # Try to get from description
        if hasattr(task, "description"):
            desc = task.description
            for phrase, task_name in self.task_description_map.items():
                if phrase.lower() in desc.lower():
                    return task_name

        # Try to get from task ID or name
        task_str = str(task)
        for task_name in self.task_sequence:
            if task_name in task_str:
                return task_name

        return "unknown_task"

    def _get_agent_name_for_task(self, task_name):
        """Get agent name for the task"""
        for agent, tasks in self.agent_task_map.items():
            if task_name in tasks:
                return agent
        return "System"

    def _get_task_description(self, task_name):
        """Get a human-readable description for the task"""
        task_descriptions = {
            "probabilities_comment": "Analyzing the implied probabilities from market data.",
            "get_economic_suggestions": "Analyzing economic data and proposing policy solutions.",
            "regional_analysis": "Regional Pragmatists analyzing the proposed solutions.",
            "academic_analysis": "Academic Balancers evaluating the proposals.",
            "central_analysis": "Central Policymakers reviewing the proposals.",
            "regional_discussion": "Regional Pragmatists discussing with other members.",
            "academic_discussion": "Academic Balancers engaging in discussion.",
            "central_discussion": "Central Policymakers synthesizing viewpoints.",
            "regional_vote": "Regional Pragmatists casting final vote.",
            "academic_vote": "Academic Balancers casting final vote.",
            "central_vote": "Central Policymakers casting final vote.",
            "other_summary": "Drafting the FOMC public statement.",
            "vote_summary": "Summarizing the final votes.",
            "prediction_summary": "Summarizing the rate predictions.",
            "summary_final": "Finalizing the simulation results.",
        }
        return task_descriptions.get(task_name, f"Executing task: {task_name}")

    def _add_to_conversation_logs(self, log_entry):
        """Add an entry to the conversation logs"""
        self.conversation_logs.append(log_entry)
        # Keep only the last 1000 entries
        if len(self.conversation_logs) > 1000:
            self.conversation_logs = self.conversation_logs[-1000:]

    async def broadcast_state(self):
        """Broadcast the current state to all connected clients"""
        await self.websocket_manager.broadcast(self.get_current_state())

    async def _process_final_results(self):
        """Process the final results of the simulation"""
        if not self.crew_results:
            return

        try:
            # Convert result to string if it isn't already
            result_str = str(self.crew_results)

            # Find the JSON object in the text
            # Look for the last occurrence of a JSON-like structure
            self.fomc_statement = result_str

            #     # Extract just the JSON part
            #     json_str = result_str[json_start : json_end + 1]

            #     # Clean up the JSON string
            #     # Remove any escaped quotes
            #     json_str = json_str.replace('\\"', '"')
            #     # Remove any double curly braces
            #     json_str = json_str.replace("{{", "{").replace("}}", "}")

            #     # Parse the JSON
            #     final_results = json.loads(json_str)

            #     # Update FOMC statement
            #     if "fomc_public_statement" in final_results:
            #         self.fomc_statement = final_results["fomc_public_statement"]

            #     # Update votes
            #     if "rate_votes" in final_results:
            #         self.votes_data = final_results["rate_votes"]

            #     # Update predictions
            #     if "rate_predictions" in final_results:
            #         self.predictions_data = final_results["rate_predictions"]

            # else:
            #     # If no JSON found, treat the entire output as the FOMC statement
            #     self.fomc_statement = result_str

            await self.broadcast_state()

        except Exception as e:
            self.error = (
                f"Error processing final results: {str(e)}\nResult:\n{result_str}"
            )
            await self.broadcast_state()

    def _format_output(self, output):
        """Format output for display"""
        if output is None:
            return "No output available"

        if isinstance(output, str):
            return output

        try:
            return str(output)
        except:
            return "Could not format output"

    def _process_task_output(self, task_name, output):
        """Process the output of specific tasks"""
        try:
            print(f"\n=== Processing Task: {task_name} ===")
            output_text = self._format_output(output)
            print(f"Raw output text: {output_text[:200]}...")  # Print first 200 chars
            index = -1  # Initialize index with invalid value
            member = "Unknown"  # Initialize member with default value

            if task_name == "other_summary":
                print("Processing other_summary task")
                self.fomc_statement = output_text

            elif task_name in ["regional_vote", "academic_vote", "central_vote"]:
                print(f"\nProcessing vote task: {task_name}")
                # Extract member type from task name
                member_type = task_name.split("_")[0].capitalize()
                print(f"Member type: {member_type}")

                if member_type == "Regional":
                    member = "Regional Pragmatist"
                    index = 0
                elif member_type == "Academic":
                    member = "Academic Balancer"
                    index = 1
                elif member_type == "Central":
                    member = "Central Policymaker"
                    index = 2
                print(f"Selected member: {member}, index: {index}")

                if index >= 0:  # Only process if we have a valid index
                    # Extract vote
                    vote_match = re.search(
                        r"INTEREST RATE VOTE:\s*([-+]?\d+\.?\d*%)", output_text
                    )
                    if vote_match:
                        vote = vote_match.group(1)
                        print(f"Found vote: {vote}")
                        self.votes_data[index]["vote"] = vote
                    else:
                        print("No vote match found in text")

                    # Extract policy vote
                    policy_match = re.search(
                        r"POLICY VOTE:\s*(.+?)(?=\n|INTEREST|$)", output_text
                    )
                    if policy_match:
                        policy = policy_match.group(1).strip()
                        print(f"Found policy vote: {policy}")
                        self.votes_data[index]["policy_vote"] = policy
                    else:
                        print("No policy vote match found in text")

                    # Extract prediction
                    prediction_match = re.search(
                        r"PREDICTION FOR 2025:\s*([-+]?\d+\.?\d*%)", output_text
                    )
                    if prediction_match:
                        prediction = prediction_match.group(1)
                        print(f"Found prediction: {prediction}")
                        self.predictions_data[index]["prediction"] = prediction
                    else:
                        print("No prediction match found in text")

                    print(f"\nUpdated data for {member}:")
                    print(f"Votes data: {self.votes_data[index]}")
                    print(f"Predictions data: {self.predictions_data[index]}")

        except Exception as e:
            print(f"\nError in _process_task_output: {str(e)}")
            print(f"Task name: {task_name}")
            print(f"Output text: {output_text[:200]}...")  # Print first 200 chars
            traceback.print_exc()


simulation_manager = SimulationManager(websocket_manager)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        await websocket.send_json(simulation_manager.get_current_state())

        while True:
            data = await websocket.receive_json()
            if data.get("action") == "start_simulation":
                asyncio.create_task(simulation_manager.run_simulation())
            elif data.get("action") == "reset_simulation":
                simulation_manager.reset()
                await websocket.send_json(simulation_manager.get_current_state())
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/api/status")
async def get_status():
    """Get the current simulation status via REST API"""
    return simulation_manager.get_current_state()


@app.post("/api/start")
async def start_simulation():
    """Start the simulation via REST API"""
    if not simulation_manager.is_running:
        asyncio.create_task(simulation_manager.run_simulation())
    return {"status": "started"}


@app.post("/api/reset")
async def reset_simulation():
    """Reset the simulation via REST API"""
    simulation_manager.reset()
    return {"status": "reset"}


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    date_prefix: str = Form(...),
):
    """
    Upload files for the FOMC simulation
    """
    try:
        # Create knowledge directory if it doesn't exist
        knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        os.makedirs(knowledge_dir, exist_ok=True)

        uploaded_files = []

        for file in files:
            # Add the date prefix to PDFs and CSVs
            if file.filename.lower().endswith(".pdf"):
                filename = f"{date_prefix} {file.filename}"
            elif file.filename.lower().endswith(".csv"):
                filename = f"{date_prefix} {file.filename}"
            else:
                filename = file.filename

            file_path = os.path.join(knowledge_dir, filename)

            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            uploaded_files.append(filename)

        return {"success": True, "files": uploaded_files}

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Failed to upload files: {str(e)}"}
        )


@app.get("/api/available-dates")
async def get_available_dates():
    """Get the list of available dates from the knowledge directory"""
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    files = os.listdir(knowledge_dir)

    # Extract unique date prefixes (format: YY_M)
    dates = set()
    for file in files:
        if file.endswith((".pdf", ".csv")):
            parts = file.split()
            if parts and len(parts[0].split("_")) == 2:  # Ensure it matches YY_M format
                dates.add(parts[0])

    # Sort dates in reverse chronological order
    sorted_dates = sorted(list(dates), reverse=True)
    return {"dates": sorted_dates}


class DateRequest(BaseModel):
    date: str


@app.post("/api/set-date")
async def set_date(request: DateRequest):
    """Set the selected date for the simulation"""
    if simulation_manager.is_running:
        return JSONResponse(
            status_code=400,
            content={"message": "Cannot change date while simulation is running"},
        )

    # Validate that the date exists in the knowledge directory
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    files = os.listdir(knowledge_dir)
    valid_dates = set()
    for file in files:
        if file.endswith((".pdf", ".csv")):
            parts = file.split()
            if parts and len(parts[0].split("_")) == 2:
                valid_dates.add(parts[0])

    if request.date not in valid_dates:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Invalid date: {request.date}. Must be one of {sorted(list(valid_dates))}"
            },
        )

    simulation_manager.selected_date = request.date
    simulation_manager.crew_instance = BnyCapstoneCrew(date=request.date)
    simulation_manager.reset()
    return {"status": "success", "date": request.date}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
