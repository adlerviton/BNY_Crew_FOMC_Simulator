from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()

### Add llm
managerllm = LLM(model="openai/gpt-4o", temperature=0.03)
# Always keep manager as GPT 4o

gpt_llm = LLM(model="openai/gpt-4o-mini", temperature=0.03)


class CallbackCrew(Crew):
    """Custom Crew class that implements callback functionality"""

    def __init__(self, agents, tasks, process, **kwargs):
        super().__init__(agents=agents, tasks=tasks, process=process, **kwargs)
        self.task_callback = None
        self.step_callback = None
        self._current_task = None

    def kickoff(self):
        """Override kickoff to implement callbacks"""
        for task in self.tasks:
            # Get the agent for this task
            agent = task.agent

            # Set the step callback on the agent
            if hasattr(agent, "step_callback"):
                agent.step_callback = self.step_callback

            # Only call task start callback if it's a different task
            if self.task_callback and task != self._current_task:
                self._current_task = task
                self.task_callback(task, output=None, is_start=True)

            # Execute the task
            result = agent.execute_task(task)

            # Call task callback for completion
            if self.task_callback:
                self.task_callback(task, output=result, is_start=False)

            if not result:
                break

        return result


@CrewBase
class BnyCapstoneCrew:
    """FOMC Simulation crew with initial member analysis"""

    agents_config = "config/agents.yaml"
    # tasks not needed as we have to manually change them
    tasks_config = "config/tasks.yaml"

    def __init__(self, date=None):
        """Initialize the crew with a specific date"""
        self.date = date or "23_6"  # Default to 23_6 if no date provided
        print(self.date)
        # Initialize knowledge sources, the pdfs need to be brute force loaded as strings
        self.pdf_sources = PDFKnowledgeSource(
            file_path=[
                os.path.join(f"{self.date} current macro 1.pdf"),
                # os.path.join(f"{self.date} dot plot description.pdf"),
                # os.path.join("Fed Explanation.pdf"),
            ]
        )
        self.bb_source = PDFKnowledgeSource(
            file_paths=[os.path.join(f"{self.date} beige book.pdf")]
        )

        self.dpd_source = PDFKnowledgeSource(
            file_paths=[os.path.join(f"{self.date} dot plot description.pdf")]
        )

        self.fed_source = PDFKnowledgeSource(
            file_paths=[os.path.join(f"Fed Explanation.pdf")]
        )

        print(self.pdf_sources)

        self.csv_source = CSVKnowledgeSource(
            file_paths=[os.path.join(f"{self.date} historical macro.csv")]
        )

    # Define agents
    @agent
    def economist(self) -> Agent:
        agent = Agent(
            config=self.agents_config["economist"],
            verbose=True,
            llm="gpt-4o",
            knowledge_sources=[self.bb_source],
            max_iter=50,
        )
        agent.step_callback = None
        return agent

    @agent
    def Regional_Pragmatists(self) -> Agent:
        agent = Agent(
            config=self.agents_config["Regional_Pragmatists"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[
                self.pdf_sources,
                self.csv_source,
                self.bb_source,
                self.dpd_source,
                self.fed_source,
            ],
            max_iter=50,
            memory=True,
        )
        agent.step_callback = None
        return agent

    @agent
    def Academic_Balancers(self) -> Agent:
        agent = Agent(
            config=self.agents_config["Academic_Balancers"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[
                self.pdf_sources,
                self.csv_source,
                self.bb_source,
                self.dpd_source,
                self.fed_source,
            ],
            max_iter=50,
            memory=True,
        )
        agent.step_callback = None
        return agent

    @agent
    def Central_Policymakers(self) -> Agent:
        agent = Agent(
            config=self.agents_config["Central_Policymakers"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[
                self.pdf_sources,
                self.csv_source,
                self.bb_source,
                self.dpd_source,
                self.fed_source,
            ],
            max_iter=50,
            memory=True,
        )
        agent.step_callback = None
        return agent

    @agent
    def analyst(self) -> Agent:
        print("analyst")
        # print(self.pdf_source)
        agent = Agent(
            config=self.agents_config["analyst"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[self.pdf_sources],
            max_iter=50,
            memory=True,
        )
        agent.step_callback = None
        print(agent.knowledge_sources)
        return agent

    # Define economist tasks
    @task
    def probabilities_comment(self) -> Task:
        print("probabilities_comment")
        print(list(self.analyst().knowledge_sources[0].content.values())[0])
        check = list(self.analyst().knowledge_sources[0].content.values())[0]
        return Task(
            description=f"""
            Make a comment about the implied probabilities in the {self.date} macro 1.pdf file.
            Explain what this means, this data comes from the CME Fedwatch website.
            Knowledge material: {check}
            """,
            agent=self.analyst(),
            expected_output=f"A comment about the implied probabilities in the current macro 1.pdf file, explaining what it means.",
        )

    @task
    def get_economic_suggestions(self) -> Task:
        # print(self.economist().knowledge_sources)
        check2 = list(self.analyst().knowledge_sources[0].content.values())[0]
        check = list(self.economist().knowledge_sources[0].content.values())[0]
        return Task(
            description=f"""
            As the Fed's chief economist, analyze the current economic situation using the Beige Book, 
            historical macro data, and current economic indicators provided to you. Pay attention to the 
            probabilities implied by the futures market, mentioned in the current macro 1.pdf file.
            
            Based on your analysis, propose 3 potential monetary policy solutions, each with:
            1. A title that includes a specific interest rate plan with numerical adjustment
            2. Detailed justification based on economic data
            3. Projected economic outcomes
            4. A reference to the implied probabilities of rate hikes and rate cuts mentioned
            in the knowledge material.
            
            You must not describe any preliminary leanings or advice on which option is best.
            
            Format your response clearly with each solution separated and numbered.
            Never give options that have a 0 percent implied probability. For example, if the 
            implied probabilities show a 0 percent chance of a rate hike, none of the options 
            should include increasing the rate. A 100% chance of a rate hike or a rate cut does
            not mean that the market favors a steady rate.
            Knowledge material: {check}
            Knowledge material 2: {check2}
            """,
            agent=self.economist(),
            context=[self.probabilities_comment()],
            expected_output="A comprehensive analysis with 3-5 detailed monetary policy solutions that include the numerical Fed Funds target rate adjustment in the title of each solution.",
        )

    # Define member analysis tasks
    @task
    def regional_analysis(self) -> Task:
        # check2 = list(
        #     self.Regional_Pragmatists().knowledge_sources[0].content.values()
        # )[0]
        # check = list(self.Regional_Pragmatists().knowledge_sources[0].content.values())[
        #     0
        # ]
        print("regional")
        # print(self.Regional_Pragmatists().knowledge_sources[0])

        macro = list(self.Regional_Pragmatists().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Regional_Pragmatists().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Regional_Pragmatists().knowledge_sources[2].content.values())[0]
        dpd = list(self.Regional_Pragmatists().knowledge_sources[3].content.values())[0]
        fed = list(self.Regional_Pragmatists().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Regional Pragmatists, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the {self.date} current macro 1.pdf file.
            
            Consider:
            1. The current economic data from the Beige Book
            2. Historical macroeconomic data
            3. Current macroeconmic data
            4. The fed statement and minutes from the prior meeting
            5. Your perspective on inflation, employment, and financial stability
            
            Provide your initial assessment of each proposed solution, highlighting strengths, 
            weaknesses, and your preliminary leanings. Focus on how these policies align with the 
            Fed's dual mandate of price stability and maximum employment. Make specific historical
            comparisons using exact dates, and reference specific macroeconomic indicators.
            
            End with your preliminary thoughts on which direction you're leaning and why.
            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Regional_Pragmatists(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Regional Pragmatists' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    @task
    def academic_analysis(self) -> Task:
        macro = list(self.Academic_Balancers().knowledge_sources[0].content.values())[0]
        csv_s = list(self.Academic_Balancers().knowledge_sources[1].content.values())[0]
        bb = list(self.Academic_Balancers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Academic_Balancers().knowledge_sources[3].content.values())[0]
        fed = list(self.Academic_Balancers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            Before you start, make sure you have read the {self.date} current macro 1.pdf file. if you cant find it, dont do the task.

            As Academic Balancers, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the current macro 1.pdf file.
            
            Consider:
            1. The current economic data from the Beige Book
            2. Historical macroeconomic data
            3. Current macroeconmic data
            4. The fed statement and minutes from the prior meeting
            5. Your perspective on inflation, employment, and financial stability
            
            Provide your initial assessment of each proposed solution, highlighting strengths, 
            weaknesses, and your preliminary leanings. Focus on how these policies align with the 
            Fed's dual mandate of price stability and maximum employment. Make specific historical
            comparisons using exact dates, and reference specific macroeconomic indicators.
            
            End with your preliminary thoughts on which direction you're leaning and why.

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Academic_Balancers(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Academic Balancers' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    @task
    def central_analysis(self) -> Task:
        macro = list(self.Central_Policymakers().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Central_Policymakers().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Central_Policymakers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Central_Policymakers().knowledge_sources[3].content.values())[0]
        fed = list(self.Central_Policymakers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Central Policymakers, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the current {self.date} macro 1.pdf file.
            
            Consider:
            1. The current economic data from the Beige Book
            2. Historical macroeconomic data
            3. Current macroeconmic data
            4. The fed statement and minutes from the prior meeting
            5. Your perspective on inflation, employment, and financial stability
            
            Provide your initial assessment of each proposed solution, highlighting strengths, 
            weaknesses, and your preliminary leanings. Focus on how these policies align with the 
            Fed's dual mandate of price stability and maximum employment. Make specific historical
            comparisons using exact dates, and reference specific macroeconomic indicators.
            
            End with your preliminary thoughts on which direction you're leaning and why.

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Central_Policymakers(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Central Policymakers' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    # Define individual discussion tasks for each member
    @task
    def regional_discussion(self) -> Task:
        macro = list(self.Regional_Pragmatists().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Regional_Pragmatists().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Regional_Pragmatists().knowledge_sources[2].content.values())[0]
        dpd = list(self.Regional_Pragmatists().knowledge_sources[3].content.values())[0]
        fed = list(self.Regional_Pragmatists().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Regional Pragmatists, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            4. Highlight regional economic considerations that may have been overlooked
            
            Be specific in your references to other members' positions. Provide your assessment
            of where the committee appears to be leaning and the key factors that should inform the final decision.
            Be sure to continue to reference specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Regional_Pragmatists(),
            context=[
                self.regional_analysis(),
                self.academic_analysis(),
                self.central_analysis(),
            ],
            expected_output="Regional Pragmatists' response to member analyses and assessment of committee direction. This includes a vote for what to do with the interest rate, specific historical comparisons to exact dates, mentions of specific metrics, and a prediction for the fed funds target rate at the end of 2025.",
        )

    @task
    def academic_discussion(self) -> Task:
        macro = list(self.Academic_Balancers().knowledge_sources[0].content.values())[0]
        csv_s = list(self.Academic_Balancers().knowledge_sources[1].content.values())[0]
        bb = list(self.Academic_Balancers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Academic_Balancers().knowledge_sources[3].content.values())[0]
        fed = list(self.Academic_Balancers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Academic Balancers, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            4. Highlight the most important considerations that should guide the committee's decision
            
            Be specific in your references to other members' positions. Be sure to continue to reference
            specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Academic_Balancers(),
            context=[
                self.regional_analysis(),
                self.academic_analysis(),
                self.central_analysis(),
            ],
            expected_output="Academic Balancers' response to member analyses. This includes a vote for what to do with the interest rate, specific historical comparisons to exact dates, mentions of specific metrics, and a prediction for the fed funds target rate at the end of 2025.",
        )

    @task
    def central_discussion(self) -> Task:
        macro = list(self.Central_Policymakers().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Central_Policymakers().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Central_Policymakers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Central_Policymakers().knowledge_sources[3].content.values())[0]
        fed = list(self.Central_Policymakers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Central Policymakers, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            
            Be specific in your references to other members' positions. Be sure to continue to reference
            specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Central_Policymakers(),
            context=[
                self.regional_analysis(),
                self.academic_analysis(),
                self.central_analysis(),
            ],
            expected_output="Central Policymakers' response to member analyses. This includes a vote for what to do with the interest rate, specific historical comparisons to exact dates, mentions of specific metrics, and a prediction for the fed funds target rate at the end of 2025.",
        )

    @task
    def regional_vote(self) -> Task:
        macro = list(self.Regional_Pragmatists().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Regional_Pragmatists().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Regional_Pragmatists().knowledge_sources[2].content.values())[0]
        dpd = list(self.Regional_Pragmatists().knowledge_sources[3].content.values())[0]
        fed = list(self.Regional_Pragmatists().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Regional Pragmatists, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range.

            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".
            
            Format your response as:
            Regional Pragmatist:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025, a specific number, not a range]

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Regional_Pragmatists(),
            context=[
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
            ],
            expected_output="Clear vote with formatted response as: Regional Pragmatist, POLICY VOTE: [Policy Option #], INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for], SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years], EXPLANATION: [Your explanation including specific metrics], PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025].",
        )

    @task
    def academic_vote(self) -> Task:
        macro = list(self.Academic_Balancers().knowledge_sources[0].content.values())[0]
        csv_s = list(self.Academic_Balancers().knowledge_sources[1].content.values())[0]
        bb = list(self.Academic_Balancers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Academic_Balancers().knowledge_sources[3].content.values())[0]
        fed = list(self.Academic_Balancers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Academic Balancers, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range. 
            
            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".

            Format your response as:
            Academic Balancer:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025, a specific number, not a range]

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Academic_Balancers(),
            context=[
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
            ],
            expected_output="Clear vote with formatted response as: Academic Balancer, POLICY VOTE: [Policy Option #], INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for], SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years], EXPLANATION: [Your explanation including specific metrics], PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025].",
        )

    @task
    def central_vote(self) -> Task:
        macro = list(self.Central_Policymakers().knowledge_sources[0].content.values())[
            0
        ]
        csv_s = list(self.Central_Policymakers().knowledge_sources[1].content.values())[
            0
        ]
        bb = list(self.Central_Policymakers().knowledge_sources[2].content.values())[0]
        dpd = list(self.Central_Policymakers().knowledge_sources[3].content.values())[0]
        fed = list(self.Central_Policymakers().knowledge_sources[4].content.values())[0]
        return Task(
            description=f"""
            As Central Policymakers, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range.

            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".
            
            Format your response as:
            Central Policymaker:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025. A specific number, not a range]

            Knowledge material currnt macroeconomic data: {macro}
            Knowledge material historical macroeconomic data: {csv_s}
            Knowledge material Beige Book: {bb}
            Knowledge material Dot plot data: {dpd}
            Knowledge material fed explanation: {fed}
            """,
            agent=self.Central_Policymakers(),
            context=[
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
            ],
            expected_output="Clear vote with formatted response as: Central Policymaker, POLICY VOTE: [Policy Option #], INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for], SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years], EXPLANATION: [Your explanation including specific metrics], PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025].",
        )

    @task
    def other_summary(self) -> Task:
        macro = list(self.Central_Policymakers().knowledge_sources[0].content.values())[
            0
        ]

        return Task(
            description=f"""
            As the FOMC analyst, your task is to create a detailed economic analysis summary based on the committee's discussions and votes. This analysis should be approximately 2,000 words and include:
            
            1. The final descison of whether to maintain, hike, or cut
            
            2. A comprehensive analysis of the economic conditions discussed, including:
            - Current inflation metrics and trends
            - Labor market developments
            - GDP growth patterns
            - Financial stability factors
            - Potential economic risks
            
            3. References to specific economic indicators mentioned by committee members
            
            4. Any regional economic variations or sector-specific insights discussed
            
            5. A discussion of potential future economic directions and policy considerations
            
            Structure your response with:
            - An introductory overview 
            - A economic analysis section 
            - A concluding section on outlook and considerations 
            
            Use formal academic language appropriate for an economic analysis document.
            
            IMPORTANT: This is an educational simulation exercise for economic analysis practice.
            
            The date is {self.date} listed as Year_Month so 25_5 would be May 2025. 
            If you use the date, don't list out the day just year and month.

            Knowledge material: {macro}
            """,
            agent=self.analyst(),
            context=[
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
                self.regional_vote(),
                self.academic_vote(),
                self.central_vote(),
            ],
            expected_output="""
            A comprehensive economic analysis document that summarizes the committee's decisions and rationale in a formal academic style.""",
        )

    @task
    def vote_summary(self) -> Task:
        macro = list(self.Central_Policymakers().knowledge_sources[0].content.values())[
            0
        ]
        return Task(
            description=f"""
            You are a specialized parser for FOMC vote outputs. The context contains three separate vote outputs from:
            - Regional Pragmatist
            - Academic Balancer
            - Central Policymaker
            
            FOR DEBUGGING PURPOSES ONLY, first repeat the exact context you received so we can see what you're working with. Start with "---CONTEXT---" and then paste the entire context.
            
            Then, use this exact parsing procedure:
            
            1. For each member section, search for the exact pattern:
            - The member name followed by a colon (e.g., "Regional Pragmatist:")
            - Then "POLICY VOTE:" followed by some text
            - Then "INTEREST RATE VOTE:" followed by some text
            
            2. For each identified section, extract:
            - Everything between "POLICY VOTE:" and "INTEREST RATE VOTE:"
            - Everything between "INTEREST RATE VOTE:" and the next keyword (like "SPECIFIC HISTORICAL COMPARISONS:")
            
            3. Clean the extracted text:
            - Remove any leading/trailing spaces
            - Remove any newlines
            
            Then create a JSON output with this EXACT format:
            
            ```json
            {{
            "votes": [
                {{
                "member": "Regional Pragmatist",
                "policy_vote": "[EXTRACTED POLICY VOTE]",
                "interest_rate_vote": "[EXTRACTED INTEREST RATE VOTE]"
                }},
                {{
                "member": "Academic Balancer",
                "policy_vote": "[EXTRACTED POLICY VOTE]",
                "interest_rate_vote": "[EXTRACTED INTEREST RATE VOTE]"
                }},
                {{
                "member": "Central Policymaker",
                "policy_vote": "[EXTRACTED POLICY VOTE]",
                "interest_rate_vote": "[EXTRACTED INTEREST RATE VOTE]"
                }}
            ]
            }}
            ```
            
            IMPORTANT: If you cannot find a section or a specific field, use the text "NOT FOUND" for that field.
            
            knowledge current macro: {macro}
            """,
            agent=self.analyst(),
            context=[
                self.regional_vote(),
                self.academic_vote(),
                self.central_vote(),
            ],
            expected_output="""
            A JSON object containing the extracted vote information from all three members.
            """,
        )

    @task
    def prediction_summary(self) -> Task:
        return Task(
            description="""
           As the FOMC analyst, prepare a JSON summary of the end of 2025 rate predictions from all three members voting tasks.
           The predictions should accurately reflect PREDICTION FOR 2025 from each respective member's voting task.
           The output must strictly follow this format:
           {{
               "rate_predictions": [
                   {{"member": "Regional Pragmatist", "prediction": "#.##%"}}, #PREDICTION FOR 2025 from regional_vote task
                   {{"member": "Academic Balancer", "prediction": "#.##%"}}, #PREDICTION FOR 2025 from academic_vote task
                   {{"member": "Central Policymaker", "prediction": "#.##%"}} #PREDICTION FOR 2025 from central_vote task
               ]
           }}

           **Important Notes:**
           - The summary should accurately reflect the final predictions from each member's voting task.
           - The JSON must be correctly formatted with no additional text, markdown, or surrounding explanations.
           """,
            agent=self.analyst(),
            context=[
                self.regional_vote(),
                self.academic_vote(),
                self.central_vote(),
            ],
            expected_output="""
           {{
               "rate_predictions": [
                   {{"member": "Regional Pragmatist", "prediction": "#.##%"}}, #PREDICTION FOR 2025 from regional_vote task
                   {{"member": "Academic Balancer", "prediction": "#.##%"}}, #PREDICTION FOR 2025 from academic_vote task
                   {{"member": "Central Policymaker", "prediction": "#.##%"}} #PREDICTION FOR 2025 from central_vote task
               ]
           }}
           """,
        )

    @task
    def summary_final(self) -> Task:
        return Task(
            description="""
           Combine the outputs of other_summary, vote_summary, and prediction_summary into one JSON object while accurately reflecting the information from the three prior summary tasks. 
           The output must strictly follow this format:
           {{
                "fomc_public_statement": "Full FOMC-style qualitative statement text goes here.",
               "rate_votes": [
                   {{"member": "Regional Pragmatist", "vote": "#.##%"}},
                   {{"member": "Academic Balancer", "vote": "#.##%"}}, 
                   {{"member": "Central Policymaker", "vote": "#.##%"}} 
               ],
               "rate_predictions": [
                   {{"member": "Regional Pragmatist", "prediction": "#.##%"}},
                   {{"member": "Academic Balancer", "prediction": "#.##%"}}, 
                   {{"member": "Central Policymaker", "prediction": "#.##%"}}
               ]
           }}
           **Important Notes:**
           - The summary should accurately reflect the information from other_summary, vote_summary, and prediction_summary.
           - The JSON must be correctly formatted with no additional text, markdown, or surrounding explanations.
           - If you can't read any of the context just return: I CAN'T READ ANYTHING
           """,
            agent=self.analyst(),
            context=[
                self.other_summary(),
                self.vote_summary(),
                self.prediction_summary(),
            ],
            expected_output="""
            {{
                "fomc_public_statement": "Full FOMC-style qualitative statement text goes here.",
                "rate_votes": [
                    {{"member": "Regional Pragmatist", "vote": "#.##%"}},
                    {{"member": "Academic Balancer", "vote": "#.##%"}}, 
                    {{"member": "Central Policymaker", "vote": "#.##%"}} 
                ],
                "rate_predictions": [
                    {{"member": "Regional Pragmatist", "prediction": "#.##%"}},
                    {{"member": "Academic Balancer", "prediction": "#.##%"}},
                    {{"member": "Central Policymaker", "prediction": "#.##%"}}
                ]
            }}
            """,
            output_file="rate_summary.json",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FOMC simulation crew"""
        print("crew")
        print(self.pdf_sources)
        return CallbackCrew(
            agents=[
                self.economist(),
                self.Regional_Pragmatists(),
                self.Academic_Balancers(),
                self.Central_Policymakers(),
                self.analyst(),
            ],
            tasks=[
                # First get economic suggestions
                self.probabilities_comment(),
                self.get_economic_suggestions(),
                # Then have each member analyze the suggestions
                self.regional_analysis(),
                self.academic_analysis(),
                self.central_analysis(),
                # Then have individual discussion contributions from each member
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
                # Then voting
                self.regional_vote(),
                self.academic_vote(),
                self.central_vote(),
                # Finally summary and statement with votes
                self.other_summary(),
            ],
            process=Process.sequential,  # Use sequential process to ensure proper order
            verbose=True,
            memory=True,
            output_log_file="fomc_simulation.md",
        )
