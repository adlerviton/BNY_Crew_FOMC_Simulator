from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai import LLM
import os
import os

### Add knowledge source, must add pdfs and csv in knoweldege folder
date = "22_6"
pdf_source = PDFKnowledgeSource(
    file_paths=[
        f"{date} beige book.pdf",
        f"{date} current macro 1.pdf",
        f"{date} dot plot description.pdf",
        "Fed Explanation.pdf",
    ]
)

csv_source = CSVKnowledgeSource(file_paths=[f"{date} historical macro.csv"])

### add llm
managerllm = LLM(model="openai/gpt-4o", temperature=0.03)
# Always keep manager as GPT 4o

gpt_llm = LLM(model="openai/gpt-4o-mini", temperature=0.03)

deepseek_llm = LLM(
    model="openrouter/deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

claude_llm = LLM(
    model="claude-3-5-sonnet-20240620",
    base_url="https://api.anthropic.com",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


@CrewBase
class BnyCapstoneCrew:
    """FOMC Simulation crew with initial member analysis"""

    agents_config = "config/agents.yaml"
    # tasks not needed as we have to manually change them
    tasks_config = "config/tasks.yaml"

    # Define agents
    @agent
    def economist(self) -> Agent:
        return Agent(
            config=self.agents_config["economist"],
            verbose=True,
            llm="gpt-4o",
            knowledge_sources=[pdf_source, csv_source],
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

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            verbose=True,
            llm=gpt_llm,
            knowledge_sources=[pdf_source, csv_source],
            max_iter=50,
            memory=True,
        )

    # Define economist tasks
    @task
    def probabilities_comment(self) -> Task:
        return Task(
            description="""
            Make a comment about the implied probabilities in the current macro 1.pdf file.
            Explain what this means, this data comes from the CME Fedwatch website.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            """,
            agent=self.analyst(),
            expected_output="A Chain of draft style comment about the implied probabilities in the current macro 1.pdf file, explaining what it means.",
        )

    @task
    def get_economic_suggestions(self) -> Task:
        return Task(
            description="""
            As the Fed's chief economist, analyze the current economic situation using the Beige Book, 
            historical macro data, and current economic indicators provided to you. Pay attention to the 
            probabilities implied by the futures market, mentioned in the current macro 1.pdf file.
            
            Based on your analysis, propose 3 potential monetary policy solutions, each with:
            1. A title that includes a specific interest rate plan with numerical adjustment
            2. Detailed justification based on economic data
            3. Projected economic outcomes
            4. A reference to the implied probabilities of rate hikes and rate cuts mentioned
            in the knowledge material.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            
            You must not describe any preliminary leanings or advice on which option is best.
            
            Format your response clearly with each solution separated and numbered.
            Never give options that have a 0 percent implied probability. For example, if the 
            implied probabilities show a 0 percent chance of a rate hike, none of the options 
            should include increasing the rate. A 100% chance of a rate hike or a rate cut does
            not mean that the market favors a steady rate.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            """,
            agent=self.economist(),
            context=[self.probabilities_comment()],
            expected_output="A comprehensive analysis with 3-5 detailed monetary policy solutions that include the numerical Fed Funds target rate adjustment in the title of each solution.",
        )

    # Define member analysis tasks
    @task
    def regional_analysis(self) -> Task:
        return Task(
            description="""
            As Regional Pragmatists, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the current macro 1.pdf file.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            
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
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            """,
            agent=self.Regional_Pragmatists(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Regional Pragmatists' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    @task
    def academic_analysis(self) -> Task:
        return Task(
            description="""
            As Academic Balancers, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the current macro 1.pdf file.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            
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
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            """,
            agent=self.Academic_Balancers(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Academic Balancers' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    @task
    def central_analysis(self) -> Task:
        return Task(
            description="""
            As Central Policymakers, analyze the economist's proposed monetary policy solutions.
            Also, make a prediction for what the Federal Funds Target rate will be at the end
            of the year in 2025. Pay attention to the probabilities implied by the futures market, 
            mentioned in the current macro 1.pdf file.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            
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
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            """,
            agent=self.Central_Policymakers(),
            context=[self.get_economic_suggestions(), self.probabilities_comment()],
            expected_output="Central Policymakers' detailed analysis of proposed solutions with initial position, specific historical comparisons to exact dates, and a prediction for the Fed Funds target rate at the end of 2025.",
        )

    # Define individual discussion tasks for each member
    @task
    def regional_discussion(self) -> Task:
        return Task(
            description="""
            As Regional Pragmatists, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            4. Highlight regional economic considerations that may have been overlooked
            
            Be specific in your references to other members' positions. Provide your assessment
            of where the committee appears to be leaning and the key factors that should inform the final decision.
            Be sure to continue to reference specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
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
        return Task(
            description="""
            As Academic Balancers, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            4. Highlight the most important considerations that should guide the committee's decision
            
            Be specific in your references to other members' positions. Be sure to continue to reference
            specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
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
        return Task(
            description="""
            As Central Policymakers, respond to the initial analyses provided by your colleagues:
            
            1. Address key points raised by other members that align with or differ from your perspective
            2. Clarify or defend your position based on questions or concerns from others
            3. Note any shifts in your thinking based on others' analyses
            
            Be specific in your references to other members' positions. Be sure to continue to reference
            specific historical comparisons, and specific macroeconomic indicators.
            Provide an updated prediction for the fed funds target rate at the end of 2025.
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
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
        return Task(
            description="""
            As Regional Pragmatists, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range.

            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.

            Format your response as:
            Regional Pragmatist:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025, a specific number, not a range]
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
        return Task(
            description="""
            As Academic Balancers, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range. 
            
            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.

            Format your response as:
            Academic Balancer:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025, a specific number, not a range]
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
        return Task(
            description="""
            As Central Policymakers, it is time to cast your final vote and prediction.
            
            1. Clearly state which policy option you are voting for
            2. Provide a concise explanation for your vote (1-2 paragraphs)
            3. Include historical comparisons referencing specific years
            4. Make sure to reference specific macroeconomic metrics/indicators
            5. If applicable, note any reservations or conditions to your support
            6. Make a prediction for the fed funds target rate at the end of 2025, a specific number, not a range.

            The votes should be in terms of the change in interest rate, meaning if the vote is to "maintain", that means "0.00%",
            a vote to increase by 25 bps means "0.25%", and a vote to deecrease by 25 bps would mean "-0.25%".
            Use Chain of Draft: Analyze with minimal steps, 30 words max per step.
            
            Format your response as:
            Central Policymaker:
            POLICY VOTE: [Policy Option #]
            INTEREST RATE VOTE: [The interest rate adjustment included in the policy option you voted for]
            SPECIFIC HISTORICAL COMPARISONS: [historical comparisons by years]
            EXPLANATION: [Your explanation including specific metrics]
            PREDICTION FOR 2025: [your final prediction for the fed funds target rate at the end of 2025. A specific number, not a range]
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
    ### change this one
    def other_summary(self) -> Task:
        return Task(
            description="""
            As the FOMC analyst, draft a **formal public statement** on behalf of the Federal Open Market Committee (FOMC), with a total length of approximately **2,000 words**.

            This statement should:
            - Clearly announce the interest rate decision made by the committee (based on the final votes).
            - Summarize the full set of economic conditions informing the decision, including inflation trends, labor market developments, GDP growth, financial stability, and risks.
            - Reference macroeconomic indicators and specific historical analogs (e.g., "similar to conditions in 1994-95") mentioned by members.
            - Include regional variations or sectoral insights if discussed.
            - Provide forward guidance on future rate policy and the outlook for inflation and employment.
            - Be written in **the same tone, structure, and formality** as official FOMC post-meeting statements.

            Formatting Instructions:
            - Begin with a 1–2 paragraph overview of the decision.
            - Follow with a deep dive into the economic data and risk assessments (~1,200–1,500 words).
            - End with detailed forward guidance and policy direction (~300–400 words).
            - Avoid headings or markdown.

            Word Count Guidance:
            - Aim for approximately 2,000 words (you may slightly exceed this if necessary).
            - If needed, use sentence expansions or insert additional nuance consistent with Fed tone.
            """,
            agent=self.analyst(),
            context=[
                self.regional_analysis(),
                self.academic_analysis(),
                self.central_analysis(),
                self.regional_discussion(),
                self.academic_discussion(),
                self.central_discussion(),
                self.regional_vote(),
                self.academic_vote(),
                self.central_vote(),
            ],
            expected_output="""
            A fully formatted FOMC post-meeting statement, ~2,000 words long, suitable for public release. 
            It includes: (1) the interest rate decision, (2) macroeconomic and financial assessment, 
            (3) forward guidance. Written in professional FOMC style with no AI mention.
            """,
        )

    @task
    def vote_summary(self) -> Task:
        return Task(
            description="""
           As the FOMC analyst, prepare a JSON summary of the final votes from all three members voting tasks.
           The votes should be in terms of the change in interest rate they voted for, accurately reflecting INTEREST RATE VOTE from each respective member's voting task.
           The output must strictly follow this format:
           {{
               "rate_votes": [
                   {{"member": "Regional Pragmatist", "vote": "#.##%"}}, #INTEREST RATE VOTE from regional_vote task
                   {{"member": "Academic Balancer", "vote": "#.##%"}}, #INTEREST RATE VOTE from academic_vote task
                   {{"member": "Central Policymaker", "vote": "#.##%"}} #INTEREST RATE VOTE from central_vote task
               ],
           Important Notes:
           - The summary should accurately reflect the final votes from each member's voting task.
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
               "rate_votes": [
                   {{"member": "Regional Pragmatist", "vote": "#.##%"}}, #INTEREST RATE VOTE from regional_vote task
                   {{"member": "Academic Balancer", "vote": "#.##%"}}, #INTEREST RATE VOTE from academic_vote task
                   {{"member": "Central Policymaker", "vote": "#.##%"}} #INTEREST RATE VOTE from central_vote task
               ]
           }}
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
        return Crew(
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
                # Finally summary and statement
                self.other_summary(),
                self.vote_summary(),
                self.prediction_summary(),
                self.summary_final(),
            ],
            process=Process.sequential,  # Use sequential process to ensure proper order
            verbose=True,
            memory=True,
            output_log_file="fomc_simulation.md",
        )
