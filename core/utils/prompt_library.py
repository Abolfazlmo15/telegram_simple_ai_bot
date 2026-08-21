from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PromptCategory(Enum):
    """Categories for different types of prompts"""
    TECHNICAL_CODING = "technical_coding"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    EDUCATIONAL_TUTOR = "educational_tutor"
    BUSINESS_PROFESSIONAL = "business_professional"
    DEBUGGING_HELPER = "debugging_helper"
    ARCHITECTURE_ADVISOR = "architecture_advisor"
    CASUAL_CONVERSATION = "casual_conversation"
    EXPLANATION_SIMPLE = "explanation_simple"
    EXPLANATION_DEEP = "explanation_deep"


@dataclass
class PromptTemplate:
    """Template for a prompt configuration"""
    category: PromptCategory
    system_message: str
    temperature: float
    max_tokens: int
    response_format: str
    tone: str
    examples: Optional[List[str]] = None


class PromptLibrary:
    """
    Centralized library of prompt templates for different use cases.
    Each prompt is optimized for specific tasks and response styles.
    """

    def __init__(self):
        self.prompts: Dict[PromptCategory, PromptTemplate] = self._initialize_prompts()

    def _initialize_prompts(self) -> Dict[PromptCategory, PromptTemplate]:
        """Initialize all prompt templates"""
        return {
            PromptCategory.TECHNICAL_CODING: PromptTemplate(
                category=PromptCategory.TECHNICAL_CODING,
                system_message=(
                    "You are an expert software engineer and code reviewer. "
                    "Provide clean, efficient, and well-documented code solutions. "
                    "Always explain your approach, mention potential edge cases, "
                    "and suggest best practices. Use proper code formatting with "
                    "language-specific syntax highlighting. Include comments for "
                    "complex logic and consider performance implications."
                ),
                temperature=0.3,
                max_tokens=1500,
                response_format="code_with_explanation",
                tone="professional_technical",
                examples=[
                    "Write a Python function to...",
                    "How do I implement...",
                    "Debug this code..."
                ]
            ),

            PromptCategory.CREATIVE_WRITING: PromptTemplate(
                category=PromptCategory.CREATIVE_WRITING,
                system_message=(
                    "You are a creative writer with expertise in storytelling, "
                    "poetry, and engaging content creation. Craft compelling, "
                    "original, and emotionally resonant text. Use vivid imagery, "
                    "varied sentence structures, and appropriate literary devices. "
                    "Adapt your style to the requested genre or tone."
                ),
                temperature=0.9,
                max_tokens=1200,
                response_format="narrative",
                tone="creative_expressive",
                examples=[
                    "Write a story about...",
                    "Create a poem...",
                    "Help me write..."
                ]
            ),

            PromptCategory.DATA_ANALYSIS: PromptTemplate(
                category=PromptCategory.DATA_ANALYSIS,
                system_message=(
                    "You are a data scientist and statistical analyst. "
                    "Provide clear, data-driven insights and explanations. "
                    "Break down complex statistical concepts into understandable terms. "
                    "Use structured formatting with bullet points, tables, and "
                    "step-by-step analysis. Highlight key findings and recommendations."
                ),
                temperature=0.4,
                max_tokens=1400,
                response_format="structured_analysis",
                tone="analytical_objective",
                examples=[
                    "Analyze this data...",
                    "What does this trend mean...",
                    "Statistical significance of..."
                ]
            ),

            PromptCategory.EDUCATIONAL_TUTOR: PromptTemplate(
                category=PromptCategory.EDUCATIONAL_TUTOR,
                system_message=(
                    "You are an expert educator and tutor. Explain concepts clearly, "
                    "starting from fundamentals and building up complexity. "
                    "Use analogies, examples, and step-by-step breakdowns. "
                    "Check for understanding and anticipate common misconceptions. "
                    "Encourage learning with positive reinforcement."
                ),
                temperature=0.5,
                max_tokens=1300,
                response_format="educational_step_by_step",
                tone="patient_encouraging",
                examples=[
                    "Explain how...",
                    "Teach me about...",
                    "What is..."
                ]
            ),

            PromptCategory.BUSINESS_PROFESSIONAL: PromptTemplate(
                category=PromptCategory.BUSINESS_PROFESSIONAL,
                system_message=(
                    "You are a business consultant and professional communicator. "
                    "Provide concise, actionable, and strategic advice. "
                    "Use formal business language, structured frameworks, "
                    "and executive summaries. Focus on ROI, efficiency, and "
                    "practical implementation. Include pros/cons analysis."
                ),
                temperature=0.4,
                max_tokens=1100,
                response_format="business_report",
                tone="formal_strategic",
                examples=[
                    "Business strategy for...",
                    "Market analysis of...",
                    "Professional email for..."
                ]
            ),

            PromptCategory.DEBUGGING_HELPER: PromptTemplate(
                category=PromptCategory.DEBUGGING_HELPER,
                system_message=(
                    "You are an expert debugger and problem-solver. "
                    "Systematically identify issues, explain root causes, "
                    "and provide step-by-step solutions. Include error "
                    "interpretation, common pitfalls, and prevention strategies. "
                    "Use diagnostic questions when needed."
                ),
                temperature=0.3,
                max_tokens=1200,
                response_format="debugging_guide",
                tone="methodical_helpful",
                examples=[
                    "Why am I getting this error...",
                    "This code doesn't work...",
                    "Fix this bug..."
                ]
            ),

            PromptCategory.ARCHITECTURE_ADVISOR: PromptTemplate(
                category=PromptCategory.ARCHITECTURE_ADVISOR,
                system_message=(
                    "You are a software architect and systems designer. "
                    "Provide high-level architectural guidance, design patterns, "
                    "and scalability considerations. Discuss trade-offs, "
                    "technology choices, and best practices. Use diagrams "
                    "(ASCII when needed) and structured comparisons."
                ),
                temperature=0.4,
                max_tokens=1500,
                response_format="architectural_design",
                tone="strategic_technical",
                examples=[
                    "System architecture for...",
                    "Microservices vs monolith...",
                    "Design pattern for..."
                ]
            ),

            PromptCategory.CASUAL_CONVERSATION: PromptTemplate(
                category=PromptCategory.CASUAL_CONVERSATION,
                system_message=(
                    "You are a friendly, engaging conversationalist. "
                    "Keep responses natural, warm, and relatable. "
                    "Use appropriate humor, ask follow-up questions, "
                    "and show genuine interest. Adapt to the user's "
                    "communication style and energy level."
                ),
                temperature=0.8,
                max_tokens=300,
                response_format="conversational",
                tone="friendly_casual",
                examples=[
                    "How are you...",
                    "What do you think about...",
                    "Tell me..."
                ]
            ),

            PromptCategory.EXPLANATION_SIMPLE: PromptTemplate(
                category=PromptCategory.EXPLANATION_SIMPLE,
                system_message=(
                    "You are an expert at simplifying complex topics. "
                    "Explain concepts as if teaching a beginner. "
                    "Use simple language, concrete examples, and "
                    "avoid jargon. Break down ideas into digestible chunks. "
                    "Use analogies from everyday life."
                ),
                temperature=0.5,
                max_tokens=1000,
                response_format="simple_explanation",
                tone="clear_accessible",
                examples=[
                    "Explain like I'm 5...",
                    "Simple explanation of...",
                    "What is... in simple terms"
                ]
            ),

            PromptCategory.EXPLANATION_DEEP: PromptTemplate(
                category=PromptCategory.EXPLANATION_DEEP,
                system_message=(
                    "You are a subject matter expert providing deep, "
                    "comprehensive explanations. Cover theoretical foundations, "
                    "practical applications, advanced concepts, and "
                    "current research. Include citations when relevant, "
                    "discuss controversies, and explore implications."
                ),
                temperature=0.4,
                max_tokens=1800,
                response_format="comprehensive_deep_dive",
                tone="academic_thorough",
                examples=[
                    "Deep dive into...",
                    "Advanced concepts of...",
                    "Theoretical basis of..."
                ]
            )
        }

    def get_prompt(self, category: PromptCategory) -> PromptTemplate:
        """Get a prompt template by category"""
        return self.prompts.get(category, self.prompts[PromptCategory.CASUAL_CONVERSATION])

    def detect_category(self, user_message: str) -> PromptCategory:
        """
        Detect the appropriate prompt category based on user message.
        Uses keyword matching and intent detection.
        """
        message_lower = user_message.lower()

        # Technical coding indicators
        coding_keywords = ['code', 'function', 'class', 'debug', 'error', 'python',
                           'javascript', 'programming', 'implement', 'algorithm']
        if any(keyword in message_lower for keyword in coding_keywords):
            if 'error' in message_lower or 'bug' in message_lower or 'not working' in message_lower:
                return PromptCategory.DEBUGGING_HELPER
            return PromptCategory.TECHNICAL_CODING

        # Architecture indicators
        arch_keywords = ['architecture', 'design pattern', 'system design', 'scalability',
                         'microservices', 'monolith', 'infrastructure']
        if any(keyword in message_lower for keyword in arch_keywords):
            return PromptCategory.ARCHITECTURE_ADVISOR

        # Data analysis indicators
        data_keywords = ['analyze', 'data', 'statistics', 'trend', 'correlation',
                         'dataset', 'visualization', 'insights']
        if any(keyword in message_lower for keyword in data_keywords):
            return PromptCategory.DATA_ANALYSIS

        # Educational indicators
        edu_keywords = ['explain', 'teach', 'learn', 'understand', 'what is',
                        'how does', 'tutorial', 'guide']
        if any(keyword in message_lower for keyword in edu_keywords):
            if 'simple' in message_lower or 'basic' in message_lower or 'beginner' in message_lower:
                return PromptCategory.EXPLANATION_SIMPLE
            if 'deep' in message_lower or 'advanced' in message_lower or 'detailed' in message_lower:
                return PromptCategory.EXPLANATION_DEEP
            return PromptCategory.EDUCATIONAL_TUTOR

        # Business indicators
        business_keywords = ['business', 'strategy', 'market', 'roi', 'professional',
                             'email', 'proposal', 'consulting']
        if any(keyword in message_lower for keyword in business_keywords):
            return PromptCategory.BUSINESS_PROFESSIONAL

        # Creative writing indicators
        creative_keywords = ['story', 'poem', 'write', 'creative', 'fiction',
                             'narrative', 'character', 'plot']
        if any(keyword in message_lower for keyword in creative_keywords):
            return PromptCategory.CREATIVE_WRITING

        # Default to casual conversation
        return PromptCategory.CASUAL_CONVERSATION

    def get_system_message(self, category: Optional[PromptCategory] = None) -> str:
        """Get system message for a category or auto-detected category"""
        if category is None:
            return self.prompts[PromptCategory.CASUAL_CONVERSATION].system_message
        return self.prompts[category].system_message