from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai import LLM
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory

# from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
import os

### Add knowledge source, must add pdfs and csv in knoweldege folder
date1 = "17_12"
pdf_source = PDFKnowledgeSource(
    file_paths=[
        f"{date1} beige book.pdf",
        f"{date1} current macro 1.pdf",
        f"{date1} dot plot description.pdf",
        "Fed Explanation.pdf",
    ]
)

csv_source = CSVKnowledgeSource(
    file_paths=[
        f"{date1} historical macro.csv",
        f"{date2} historical macro.csv",
        "fedwatch.csv",
    ]
)

### add llm
managerllm = LLM(model="openai/gpt-4o", temperature=0.03)
# Always keep manager as GPT 4o

gpt_llm = LLM(model="openai/gpt-4o-mini", temperature=0.03)


@CrewBase
class BnyCapstoneCrew:
    """BnyCapstone crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Define agents
    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],
            verbose=True,
            llm=gpt_llm,
            max_iter=50,
        )

    @agent
    def Regional_Pragmatists(self) -> Agent:
        return Agent(
            config=self.agents_config["Regional_Pragmatists"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[pdf_source, csv_source],
            max_iter=50,
            memory=True,
        )

    @agent
    def Academic_Balancers(self) -> Agent:
        return Agent(
            config=self.agents_config["Academic_Balancers"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[pdf_source, csv_source],
            max_iter=50,
            memory=True,
        )

    @agent
    def Central_Policymakers(self) -> Agent:
        return Agent(
            config=self.agents_config["Central_Policymakers"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[pdf_source, csv_source],
            max_iter=50,
            memory=True,
        )

    @task
    def simulation_round1(self) -> Task:
        return Task(
            config=self.tasks_config["simulation_round1"],
        )

    @task
    def reveal_and_learn1(self) -> Task:
        return Task(
            config=self.tasks_config["reveal_and_learn1"],
        )

    @task
    def simulation_round2(self) -> Task:
        return Task(
            config=self.tasks_config["simulation_round2"],
        )

    @task
    def reveal_and_learn2(self) -> Task:
        return Task(
            config=self.tasks_config["reveal_and_learn2"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BnyCapstone crew"""
        return Crew(
            manager_agent=self.manager(),
            agents=[
                self.Regional_Pragmatists(),
                self.Academic_Balancers(),
                self.Central_Policymakers(),
            ],
            tasks=[
                self.simulation_round1(),  # New simulation round before real discussion
                self.reveal_and_learn1(),
            ],
            process=Process.hierarchical,  # Hierarchical process
            verbose=True,
            memory=True,
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path="memory/fomc_longterm.db")
            ),
            output_log_file="discussion.md",
        )
