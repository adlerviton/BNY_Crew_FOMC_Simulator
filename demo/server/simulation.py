# """
# SimulationManager:
# Handles task execution, state management, vote processing, and client communication.
# The simulation follows a predefined sequence of tasks, tracking progress and results
# through a websocket connection to the frontend. Each simulation run creates fresh
# agent instances and maintains isolated state.
# """

# import asyncio
# import json
# import time
# import os
# import sys
# import traceback
# import threading
# from typing import Dict, List, Any, Optional
# import re

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from backend.bny_capstone_crew import BnyCapstoneCrew


# class SimulationManager:
#     def __init__(self, websocket_manager):
#         self.websocket_manager = websocket_manager
#         self.crew_instance = BnyCapstoneCrew()
#         self.is_running = False
#         self.current_task = None
#         self.tasks_completed = []
#         self.crew_results = {}
#         self.error = None
#         self.progress = 0

#         # Define the task sequence for tracking progress
#         self.task_sequence = [
#             "probabilities_comment",
#             "get_economic_suggestions",
#             "regional_analysis",
#             "academic_analysis",
#             "central_analysis",
#             "regional_discussion",
#             "academic_discussion",
#             "central_discussion",
#             "regional_vote",
#             "academic_vote",
#             "central_vote",
#             "other_summary",
#             "vote_summary",
#             "prediction_summary",
#             "summary_final",
#         ]

#         # Initialize votes and predictions data
#         self.votes_data = [
#             {"member": "Regional Pragmatist", "vote": "-9999.00%"},
#             {"member": "Academic Balancer", "vote": "-9999.00%"},
#             {"member": "Central Policymaker", "vote": "-9999.00%"},
#         ]

#         self.predictions_data = [
#             {"member": "Regional Pragmatist", "prediction": "0.00%"},
#             {"member": "Academic Balancer", "prediction": "0.00%"},
#             {"member": "Central Policymaker", "prediction": "0.00%"},
#         ]

#         self.fomc_statement = ""
#         self.conversation_logs = []

#     def reset(self):
#         """Reset the simulation state"""
#         self.is_running = False
#         self.current_task = None
#         self.tasks_completed = []
#         self.crew_results = {}
#         self.error = None
#         self.progress = 0

#         self.votes_data = [
#             {"member": "Regional Pragmatist", "vote": "-9999.00%"},
#             {"member": "Academic Balancer", "vote": "-9999.00%"},
#             {"member": "Central Policymaker", "vote": "-9999.00%"},
#         ]

#         self.predictions_data = [
#             {"member": "Regional Pragmatist", "prediction": "0.00%"},
#             {"member": "Academic Balancer", "prediction": "0.00%"},
#             {"member": "Central Policymaker", "prediction": "0.00%"},
#         ]

#         self.fomc_statement = ""
#         self.conversation_logs = []

#     def get_current_state(self):
#         """Get the current state of the simulation for sending to clients"""
#         return {
#             "status": "running" if self.is_running else "idle",
#             "currentTask": self.current_task,
#             "tasksCompleted": self.tasks_completed,
#             "progress": self.progress,
#             "votes": self.votes_data,
#             "predictions": self.predictions_data,
#             "fomcStatement": self.fomc_statement,
#             "error": self.error,
#             "conversationLogs": self.conversation_logs,
#         }

#     async def broadcast_state(self):
#         """Broadcast the current state to all connected clients"""
#         await self.websocket_manager.broadcast(self.get_current_state())

#     def _broadcast_state_sync(self):
#         """Helper to broadcast state in a synchronous context"""
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             loop.run_until_complete(self.broadcast_state())
#         finally:
#             loop.close()

#     def _add_to_conversation_logs(self, log_entry):
#         """Add an entry to the conversation logs"""
#         self.conversation_logs.append(log_entry)
#         if len(self.conversation_logs) > 100:
#             self.conversation_logs = self.conversation_logs[-100:]

#     def _format_output(self, output):
#         """Format output for display"""
#         if output is None:
#             return "No output available"

#         if isinstance(output, str):
#             return output

#         try:
#             return str(output)
#         except:
#             return "Could not format output"

#     def task_callback(self, *args, **kwargs):
#         """Callback function for task execution"""
#         try:
#             task_output = args[0] if len(args) >= 1 else None

#             task_name = (
#                 self.task_sequence[len(self.tasks_completed)]
#                 if len(self.tasks_completed) < len(self.task_sequence)
#                 else None
#             )

#             if task_name and task_name not in self.tasks_completed:
#                 self._add_to_conversation_logs(
#                     {
#                         "type": "task_start",
#                         "task": task_name,
#                         "timestamp": time.time(),
#                     }
#                 )

#                 # Update progress
#                 self.progress = int(
#                     (len(self.tasks_completed) + 1) / len(self.task_sequence) * 100
#                 )

#                 # Process output
#                 output_text = self._format_output(task_output)

#                 # Log task completion
#                 self._add_to_conversation_logs(
#                     {
#                         "type": "task_complete",
#                         "task": task_name,
#                         "timestamp": time.time(),
#                         "content": output_text,
#                     }
#                 )

#                 # Add to completed tasks
#                 self.tasks_completed.append(task_name)

#                 # Process specific task outputs (votes, predictions, statement)
#                 self._process_task_output(task_name, task_output)

#                 # Broadcast state update
#                 threading.Thread(target=self._broadcast_state_sync).start()

#             return task_output

#         except Exception as e:
#             print(f"Error in task_callback: {str(e)}")
#             traceback.print_exc()
#             return None

#     def _process_task_output(self, task_name, output):
#         """Process the output of specific tasks"""
#         try:
#             output_text = self._format_output(output)

#             if task_name == "other_summary":
#                 # Try to parse the JSON output
#                 try:
#                     # Find the JSON object in the output
#                     json_start = output_text.find('{')
#                     if json_start != -1:
#                         json_str = output_text[json_start:]
#                         data = json.loads(json_str)
#
#                         # Update votes
#                         if "rate_votes" in data:
#                             self.votes_data = data["rate_votes"]
#
#                         # Update predictions
#                         if "rate_predictions" in data:
#                             self.predictions_data = data["rate_predictions"]
#
#                         # Update FOMC statement
#                         if "fomc_public_statement" in data:
#                             self.fomc_statement = data["fomc_public_statement"]
#                 except Exception as e:
#                     print(f"Error processing other_summary output: {str(e)}")
#                     traceback.print_exc()

#             elif task_name == "regional_vote":
#                 self._process_vote(output_text, 0, "Regional Pragmatist")
#             elif task_name == "academic_vote":
#                 self._process_vote(output_text, 1, "Academic Balancer")
#             elif task_name == "central_vote":
#                 self._process_vote(output_text, 2, "Central Policymaker")

#         except Exception as e:
#             print(f"Error in _process_task_output: {str(e)}")
#             traceback.print_exc()

#     def _process_vote(self, output_text, index, member_name):
#         """Process vote output"""
#         try:

#             vote_match = re.search(r"([-+]?\d+\.?\d*\s*%)", output_text)
#             if vote_match:
#                 self.votes_data[index]["vote"] = vote_match.group(1)

#             prediction_match = re.search(
#                 r"Prediction.*?([-+]?\d+\.?\d*\s*%)", output_text
#             )
#             if prediction_match:
#                 self.predictions_data[index]["prediction"] = prediction_match.group(1)

#         except Exception as e:
#             print(f"Error processing vote: {str(e)}")
#             traceback.print_exc()

#     async def run_simulation(self):
#         """Run the CrewAI simulation"""
#         if self.is_running:
#             return

#         self.is_running = True
#         self.error = None

#         # Reset the simulation manager state
#         self.reset()

#         # Create a new crew instance to ensure fresh agent memories
#         self.crew_instance = BnyCapstoneCrew()

#         await self.broadcast_state()

#         try:
#             crew = self.crew_instance.crew()
#             crew.task_callback = self.task_callback

#             def run_crew():
#                 try:
#                     result = crew.kickoff()
#                     self.crew_results = result
#                     self.is_running = False
#                     self.progress = 100

#                     if os.path.exists("rate_summary.json"):
#                         with open("rate_summary.json", "r") as f:
#                             data = json.load(f)
#                             if "rate_votes" in data:
#                                 self.votes_data = data["rate_votes"]
#                             if "rate_predictions" in data:
#                                 self.predictions_data = data["rate_predictions"]
#                             if "fomc_public_statement" in data:
#                                 self.fomc_statement = data["fomc_public_statement"]

#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)
#                     try:
#                         loop.run_until_complete(self.broadcast_state())
#                     finally:
#                         loop.close()

#                 except Exception as e:
#                     self.error = str(e)
#                     self.is_running = False
#                     print(f"Error in run_crew: {str(e)}")
#                     traceback.print_exc()

#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)
#                     try:
#                         loop.run_until_complete(self.broadcast_state())
#                     finally:
#                         loop.close()

#             thread = threading.Thread(target=run_crew)
#             thread.daemon = True
#             thread.start()

#         except Exception as e:
#             self.error = str(e)
#             self.is_running = False
#             print(f"Error starting simulation: {str(e)}")
#             traceback.print_exc()
#             await self.broadcast_state()
