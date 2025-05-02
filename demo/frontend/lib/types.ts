// frontend/lib/types.ts

export interface VoteData {
    member: string;
    vote: string;
    prediction?: string;

}

export interface PredictionData {
    member: string;
    prediction: string;
}

export interface ConversationEntry {
    type: string;
    agent?: string;
    task?: string;
    content?: string;
    timestamp: number;
    description?: string;
    vote?: string;
    prediction?: string;
}

export interface SimulationState {
    status: 'idle' | 'running' | 'completed' | 'error';
    currentTask: string | null;
    tasksCompleted: string[];
    progress: number;
    votes: VoteData[];
    predictions: PredictionData[];
    fomcStatement: string;
    error: string | null;
    conversationLogs: ConversationEntry[];
}

export const initialSimulationState: SimulationState = {
    status: 'idle',
    currentTask: null,
    tasksCompleted: [],
    progress: 0,
    votes: [
        { member: "Regional Pragmatist", vote: "0.00%" },
        { member: "Academic Balancer", vote: "0.00%" },
        { member: "Central Policymaker", vote: "0.00%" }
    ],
    predictions: [
        { member: "Regional Pragmatist", prediction: "0.00%" },
        { member: "Academic Balancer", prediction: "0.00%" },
        { member: "Central Policymaker", prediction: "0.00%" }
    ],
    fomcStatement: '',
    error: null,
    conversationLogs: []
};

export const taskNames: Record<string, string> = {
    "probabilities_comment": "Analyzing Market Probabilities",
    "get_economic_suggestions": "Generating Economic Policy Suggestions",
    "regional_analysis": "Regional Pragmatists Analysis",
    "academic_analysis": "Academic Balancers Analysis",
    "central_analysis": "Central Policymakers Analysis",
    "regional_discussion": "Regional Pragmatists Discussion",
    "academic_discussion": "Academic Balancers Discussion",
    "central_discussion": "Central Policymakers Discussion",
    "regional_vote": "Regional Pragmatists Voting",
    "academic_vote": "Academic Balancers Voting",
    "central_vote": "Central Policymakers Voting",
    "other_summary": "Preparing FOMC Statement",
    "vote_summary": "Summarizing Votes",
    "prediction_summary": "Summarizing Predictions",
    "summary_final": "Finalizing Results"
};

export const memberColors: Record<string, string> = {
    "Regional Pragmatist": "#2196f3",
    "Academic Balancer": "#4caf50",
    "Central Policymaker": "#f44336"
};