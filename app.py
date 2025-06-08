import os
import openai
import streamlit as st
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import pandas as pd

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Load environment variables
load_dotenv(override=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API Key not configured. Please add OPENAI_API_KEY to your .env file.")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class UserProfile:
    name: str
    target_language: str
    native_language: str
    current_level: str  # Beginner, Elementary, Intermediate, Upper-Intermediate, Advanced
    learning_goals: List[str]
    time_availability: str  # hours per week
    preferred_learning_style: str  # Visual, Auditory, Kinesthetic, Reading/Writing
    specific_interests: List[str]
    timeline: str  # weeks/months to achieve goals

@dataclass
class LearningPlan:
    user_profile: UserProfile
    weekly_schedule: Dict[str, List[str]]
    milestones: List[Dict[str, str]]
    resources: List[Dict[str, str]]
    assessment_schedule: List[Dict[str, str]]
    daily_activities: Dict[str, List[str]]
    estimated_timeline: str
    success_metrics: List[str]

# ============================================================================
# MULTI-AGENT SYSTEM
# ============================================================================

class LanguageLearningAgent:
    """Base class for all language learning agents"""
    
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
    
    async def generate_response(self, user_input: str, context: Dict = None) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Context: {json.dumps(context, indent=2)}"
                })
            
            messages.append({"role": "user", "content": user_input})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using the latest efficient model
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response from {self.name}: {str(e)}")
            return f"Sorry, I encountered an error. Please try again."

class ProfileAnalyzer(LanguageLearningAgent):
    """Agent specialized in analyzing user profiles and learning needs"""
    
    def __init__(self):
        super().__init__(
            name="Profile Analyzer",
            role="User Assessment Specialist",
            system_prompt="""
            You are an expert language learning consultant specializing in personalized assessment.
            
            Your role is to:
            1. Analyze user profiles comprehensively
            2. Identify learning strengths and challenges
            3. Recommend optimal learning approaches
            4. Suggest realistic timelines and goals
            
            Consider factors like:
            - Current proficiency level
            - Available time commitment
            - Learning style preferences
            - Specific goals and motivations
            - Cultural and contextual needs
            
            Provide detailed, actionable insights that other agents can use to create effective learning plans.
            """
        )
    
    async def analyze_profile(self, profile: UserProfile) -> Dict:
        """Analyze user profile and provide insights"""
        profile_text = f"""
        User Profile Analysis Request:
        
        Name: {profile.name}
        Target Language: {profile.target_language}
        Native Language: {profile.native_language}
        Current Level: {profile.current_level}
        Learning Goals: {', '.join(profile.learning_goals)}
        Time Availability: {profile.time_availability}
        Learning Style: {profile.preferred_learning_style}
        Interests: {', '.join(profile.specific_interests)}
        Timeline: {profile.timeline}
        
        Please provide:
        1. Learning readiness assessment
        2. Recommended focus areas
        3. Potential challenges to address
        4. Optimal learning strategies
        5. Realistic timeline estimation
        """
        
        analysis = await self.generate_response(profile_text)
        
        return {
            "agent": self.name,
            "analysis": analysis,
            "recommendations": self._extract_recommendations(analysis)
        }
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract key recommendations from analysis"""
        # Simple extraction - in production, you might use more sophisticated NLP
        lines = analysis.split('\n')
        recommendations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'focus on']):
                recommendations.append(line.strip())
        return recommendations[:5]  # Top 5 recommendations

class CurriculumDesigner(LanguageLearningAgent):
    """Agent specialized in creating structured learning curricula"""
    
    def __init__(self):
        super().__init__(
            name="Curriculum Designer",
            role="Educational Structure Specialist",
            system_prompt="""
            You are an expert curriculum designer for language learning programs.
            
            Your role is to:
            1. Create structured, progressive learning paths
            2. Design weekly and daily learning schedules
            3. Balance different skill areas (speaking, listening, reading, writing)
            4. Incorporate various learning activities and methods
            5. Ensure proper pacing and difficulty progression
            
            Consider:
            - CEFR levels and progression standards
            - Balanced skill development
            - Spaced repetition principles
            - Motivational factors and variety
            - Assessment and milestone integration
            
            Create practical, achievable curricula that lead to measurable progress.
            """
        )
    
    async def design_curriculum(self, profile: UserProfile, analysis: Dict) -> Dict:
        """Design a comprehensive curriculum based on profile and analysis"""
        
        curriculum_request = f"""
        Design a comprehensive language learning curriculum based on:
        
        User Profile: {asdict(profile)}
        
        Analysis Insights: {analysis.get('analysis', '')}
        
        Create:
        1. Weekly learning schedule (specific hours and activities)
        2. Daily activity recommendations
        3. Monthly milestones and goals
        4. Resource recommendations (apps, books, websites, etc.)
        5. Assessment schedule
        6. Progress tracking methods
        
        Format the response as a structured plan that can be easily followed.
        """
        
        curriculum = await self.generate_response(curriculum_request, context=asdict(profile))
        
        return {
            "agent": self.name,
            "curriculum": curriculum,
            "structured_plan": self._structure_curriculum(curriculum)
        }
    
    def _structure_curriculum(self, curriculum_text: str) -> Dict:
        """Structure the curriculum into organized components"""
        # This is a simplified version - you could enhance with better parsing
        return {
            "daily_activities": ["Practice vocabulary (15 min)", "Listen to podcasts (20 min)", "Grammar exercises (15 min)"],
            "weekly_schedule": {
                "Monday": ["Vocabulary review", "Speaking practice"],
                "Tuesday": ["Grammar focus", "Reading comprehension"],
                "Wednesday": ["Listening exercises", "Pronunciation"],
                "Thursday": ["Writing practice", "Cultural content"],
                "Friday": ["Review and assessment", "Conversation practice"],
                "Weekend": ["Immersion activities", "Fun content consumption"]
            },
            "resources": [
                {"type": "App", "name": "Recommended language app", "usage": "Daily vocabulary practice"},
                {"type": "Website", "name": "Grammar resource", "usage": "Weekly grammar lessons"},
                {"type": "Content", "name": "Podcasts/Videos", "usage": "Listening practice"}
            ]
        }

class MotivationCoach(LanguageLearningAgent):
    """Agent specialized in maintaining motivation and engagement"""
    
    def __init__(self):
        super().__init__(
            name="Motivation Coach",
            role="Engagement and Motivation Specialist",
            system_prompt="""
            You are an expert motivation coach specializing in language learning psychology.
            
            Your role is to:
            1. Create motivational strategies and techniques
            2. Design engagement activities and challenges
            3. Provide encouragement and progress celebration ideas
            4. Suggest ways to overcome learning plateaus
            5. Connect learning to personal interests and goals
            
            Focus on:
            - Intrinsic motivation development
            - Habit formation strategies
            - Progress visualization
            - Community and social learning
            - Gamification elements
            - Overcoming common obstacles
            
            Provide practical, psychology-backed strategies to maintain long-term engagement.
            """
        )
    
    async def create_motivation_strategy(self, profile: UserProfile, curriculum: Dict) -> Dict:
        """Create personalized motivation and engagement strategies"""
        
        motivation_request = f"""
        Create a comprehensive motivation strategy for:
        
        Learner Profile: {asdict(profile)}
        Learning Plan Context: {curriculum.get('structured_plan', {})}
        
        Provide:
        1. Daily motivation techniques
        2. Weekly engagement activities
        3. Progress tracking and celebration methods
        4. Strategies for overcoming common obstacles
        5. Ways to connect learning to personal interests
        6. Community and social learning suggestions
        7. Gamification elements
        8. Long-term motivation maintenance
        
        Make it personal and actionable.
        """
        
        strategy = await self.generate_response(motivation_request, context=asdict(profile))
        
        return {
            "agent": self.name,
            "strategy": strategy,
            "actionable_tips": self._extract_actionable_tips(strategy)
        }
    
    def _extract_actionable_tips(self, strategy_text: str) -> List[str]:
        """Extract actionable motivation tips"""
        lines = strategy_text.split('\n')
        tips = []
        for line in lines:
            if any(indicator in line for indicator in ['•', '-', '1.', '2.', '3.', 'Tip:', 'Try:']):
                clean_line = line.strip().lstrip('•-1234567890. ')
                if len(clean_line) > 10:  # Filter out very short lines
                    tips.append(clean_line)
        return tips[:8]  # Top 8 actionable tips

class ResourceCurator(LanguageLearningAgent):
    """Agent specialized in finding and recommending learning resources"""
    
    def __init__(self):
        super().__init__(
            name="Resource Curator",
            role="Learning Resource Specialist",
            system_prompt="""
            You are an expert at finding and recommending language learning resources.
            
            Your role is to:
            1. Recommend specific apps, websites, books, and tools
            2. Suggest content based on interests and proficiency level
            3. Provide free and paid resource options
            4. Match resources to specific learning goals
            5. Recommend authentic materials (news, podcasts, videos)
            
            Consider:
            - Quality and effectiveness of resources
            - Appropriate difficulty levels
            - Cost considerations
            - Accessibility and availability
            - Integration with learning plans
            - User interests and preferences
            
            Provide specific, actionable resource recommendations with clear usage instructions.
            """
        )
    
    async def curate_resources(self, profile: UserProfile, curriculum: Dict) -> Dict:
        """Curate personalized learning resources"""
        
        resource_request = f"""
        Recommend specific learning resources for:
        
        Language: {profile.target_language}
        Level: {profile.current_level}
        Goals: {', '.join(profile.learning_goals)}
        Interests: {', '.join(profile.specific_interests)}
        Learning Style: {profile.preferred_learning_style}
        Time Available: {profile.time_availability}
        
        Provide specific recommendations for:
        1. Mobile apps (free and paid)
        2. Websites and online platforms
        3. Books and textbooks
        4. Podcasts and audio content
        5. YouTube channels and video content
        6. Games and interactive tools
        7. Community and practice platforms
        8. Assessment tools
        
        For each resource, include:
        - Name and brief description
        - Best use case
        - Cost (free/paid)
        - How it fits into the learning plan
        """
        
        resources = await self.generate_response(resource_request, context=asdict(profile))
        
        return {
            "agent": self.name,
            "resources": resources,
            "categorized_resources": self._categorize_resources(resources)
        }
    
    def _categorize_resources(self, resources_text: str) -> Dict:
        """Categorize resources by type"""
        return {
            "apps": ["Duolingo", "Babbel", "Anki"],
            "websites": ["Language learning websites", "Grammar resources"],
            "content": ["Podcasts", "YouTube channels", "News sites"],
            "books": ["Textbooks", "Readers", "Grammar guides"],
            "tools": ["Flashcard apps", "Translation tools", "Assessment platforms"]
        }

# ============================================================================
# MULTI-AGENT COORDINATOR
# ============================================================================

class LanguageLearningCoordinator:
    """Coordinates all agents to create comprehensive learning plans"""
    
    def __init__(self):
        self.profile_analyzer = ProfileAnalyzer()
        self.curriculum_designer = CurriculumDesigner()
        self.motivation_coach = MotivationCoach()
        self.resource_curator = ResourceCurator()
    
    async def create_comprehensive_plan(self, profile: UserProfile) -> LearningPlan:
        """Coordinate all agents to create a comprehensive learning plan"""
        
        st.info("🔍 Analyzing your profile...")
        
        # Step 1: Analyze user profile
        analysis = await self.profile_analyzer.analyze_profile(profile)
        
        st.info("📚 Designing your curriculum...")
        
        # Step 2: Design curriculum
        curriculum = await self.curriculum_designer.design_curriculum(profile, analysis)
        
        st.info("💪 Creating motivation strategies...")
        
        # Step 3: Create motivation strategy
        motivation = await self.motivation_coach.create_motivation_strategy(profile, curriculum)
        
        st.info("🔗 Curating learning resources...")
        
        # Step 4: Curate resources
        resources = await self.resource_curator.curate_resources(profile, curriculum)
        
        st.success("✅ Your personalized learning plan is ready!")
        
        # Combine all results into a comprehensive plan
        learning_plan = LearningPlan(
            user_profile=profile,
            weekly_schedule=curriculum.get('structured_plan', {}).get('weekly_schedule', {}),
            milestones=[
                {"week": "Week 4", "goal": "Complete basic vocabulary set"},
                {"week": "Week 8", "goal": "Hold 5-minute conversation"},
                {"week": "Week 12", "goal": "Read simple texts"},
            ],
            resources=resources.get('categorized_resources', {}).get('apps', []),
            assessment_schedule=[
                {"frequency": "Weekly", "type": "Vocabulary quiz"},
                {"frequency": "Monthly", "type": "Speaking assessment"},
            ],
            daily_activities=curriculum.get('structured_plan', {}).get('daily_activities', []),
            estimated_timeline=f"{profile.timeline} based on {profile.time_availability} commitment",
            success_metrics=motivation.get('actionable_tips', [])[:3]
        )
        
        return learning_plan, {
            'analysis': analysis,
            'curriculum': curriculum,
            'motivation': motivation,
            'resources': resources
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'learning_plan' not in st.session_state:
        st.session_state.learning_plan = None
    if 'agent_responses' not in st.session_state:
        st.session_state.agent_responses = None

def render_profile_form():
    """Render the user profile input form"""
    st.header("📝 Create Your Learning Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name", placeholder="Enter your name")
            target_language = st.selectbox(
                "Target Language",
                ["Spanish", "French", "German", "Italian", "Portuguese", "Japanese", 
                 "Korean", "Chinese (Mandarin)", "Arabic", "Russian", "Dutch", "Other"]
            )
            current_level = st.selectbox(
                "Current Level",
                ["Complete Beginner", "Elementary (A1)", "Pre-Intermediate (A2)", 
                 "Intermediate (B1)", "Upper-Intermediate (B2)", "Advanced (C1)", "Proficient (C2)"]
            )
            time_availability = st.selectbox(
                "Time Available per Week",
                ["Less than 2 hours", "2-5 hours", "5-10 hours", "10-15 hours", "More than 15 hours"]
            )
        
        with col2:
            native_language = st.text_input("Native Language", value="English")
            learning_goals = st.multiselect(
                "Learning Goals",
                ["Conversational fluency", "Business communication", "Travel preparation",
                 "Academic study", "Cultural understanding", "Career advancement",
                 "Personal enrichment", "Family/relationship", "Immigration/relocation"]
            )
            learning_style = st.selectbox(
                "Preferred Learning Style",
                ["Visual (reading, flashcards, charts)", "Auditory (listening, speaking, music)",
                 "Kinesthetic (hands-on, movement)", "Reading/Writing (text-based)",
                 "Mixed (combination of styles)"]
            )
            timeline = st.selectbox(
                "Goal Timeline",
                ["3 months", "6 months", "1 year", "2 years", "Flexible/ongoing"]
            )
        
        interests = st.multiselect(
            "Specific Interests (helps personalize content)",
            ["Movies & TV", "Music", "Sports", "Cooking", "Travel", "Business", 
             "Technology", "Arts & Culture", "History", "Science", "Literature", 
             "News & Politics", "Fashion", "Gaming", "Social Media"]
        )
        
        submitted = st.form_submit_button("🚀 Create My Learning Plan", use_container_width=True)
        
        if submitted:
            if not all([name, target_language, current_level, learning_goals]):
                st.error("Please fill in all required fields!")
                return None
            
            profile = UserProfile(
                name=name,
                target_language=target_language,
                native_language=native_language,
                current_level=current_level,
                learning_goals=learning_goals,
                time_availability=time_availability,
                preferred_learning_style=learning_style,
                specific_interests=interests,
                timeline=timeline
            )
            
            return profile
    
    return None

def render_learning_plan(plan: LearningPlan, agent_responses: Dict):
    """Render the comprehensive learning plan"""
    st.header(f"🎯 Your Personalized {plan.user_profile.target_language} Learning Plan")
    
    # Overview
    st.subheader("📊 Plan Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Language", plan.user_profile.target_language)
        st.metric("Current Level", plan.user_profile.current_level)
    with col2:
        st.metric("Weekly Commitment", plan.user_profile.time_availability)
        st.metric("Timeline", plan.estimated_timeline)
    with col3:
        st.metric("Learning Goals", len(plan.user_profile.learning_goals))
        st.metric("Success Metrics", len(plan.success_metrics))
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Weekly Schedule", "🎯 Goals & Milestones", "📚 Resources", 
        "💪 Motivation", "🤖 Agent Insights"
    ])
    
    with tab1:
        st.subheader("Your Weekly Learning Schedule")
        
        # Daily activities
        st.write("**Daily Activities (Every Day):**")
        for activity in plan.daily_activities:
            st.write(f"• {activity}")
        
        st.write("**Weekly Schedule:**")
        for day, activities in plan.weekly_schedule.items():
            with st.expander(f"📅 {day}"):
                for activity in activities:
                    st.write(f"• {activity}")
    
    with tab2:
        st.subheader("Goals and Milestones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Learning Goals:**")
            for goal in plan.user_profile.learning_goals:
                st.write(f"🎯 {goal}")
        
        with col2:
            st.write("**Milestones:**")
            for milestone in plan.milestones:
                st.write(f"📍 {milestone.get('week', 'TBD')}: {milestone.get('goal', 'Goal TBD')}")
        
        st.write("**Success Metrics:**")
        for metric in plan.success_metrics:
            st.write(f"✅ {metric}")
    
    with tab3:
        st.subheader("Recommended Learning Resources")
        
        # Display categorized resources from the resource curator
        resources_data = agent_responses.get('resources', {})
        if resources_data.get('categorized_resources'):
            cats = resources_data['categorized_resources']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📱 Apps & Tools:**")
                for resource in cats.get('apps', []):
                    st.write(f"• {resource}")
                
                st.write("**🌐 Websites:**")
                for resource in cats.get('websites', []):
                    st.write(f"• {resource}")
            
            with col2:
                st.write("**🎵 Content & Media:**")
                for resource in cats.get('content', []):
                    st.write(f"• {resource}")
                
                st.write("**📖 Books & Materials:**")
                for resource in cats.get('books', []):
                    st.write(f"• {resource}")
        
        # Detailed resource recommendations
        if resources_data.get('resources'):
            with st.expander("📝 Detailed Resource Recommendations"):
                st.write(resources_data['resources'])
    
    with tab4:
        st.subheader("Motivation & Engagement Strategies")
        
        motivation_data = agent_responses.get('motivation', {})
        
        if motivation_data.get('actionable_tips'):
            st.write("**Daily Motivation Tips:**")
            for tip in motivation_data['actionable_tips']:
                st.write(f"💡 {tip}")
        
        if motivation_data.get('strategy'):
            with st.expander("📝 Complete Motivation Strategy"):
                st.write(motivation_data['strategy'])
    
    with tab5:
        st.subheader("AI Agent Analysis & Recommendations")
        
        # Profile Analysis
        analysis_data = agent_responses.get('analysis', {})
        if analysis_data.get('analysis'):
            with st.expander("🔍 Profile Analyzer Insights"):
                st.write(analysis_data['analysis'])
        
        # Curriculum Design
        curriculum_data = agent_responses.get('curriculum', {})
        if curriculum_data.get('curriculum'):
            with st.expander("📚 Curriculum Designer Recommendations"):
                st.write(curriculum_data['curriculum'])

def render_sidebar():
    """Render sidebar with additional features"""
    st.sidebar.title("🌟 Language Learning Hub")
    
    # Quick stats
    if st.session_state.user_profile:
        st.sidebar.success(f"👋 Hello, {st.session_state.user_profile.name}!")
        st.sidebar.info(f"🎯 Learning: {st.session_state.user_profile.target_language}")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Quick Actions")
    
    if st.sidebar.button("🔄 Create New Plan"):
        st.session_state.user_profile = None
        st.session_state.learning_plan = None
        st.session_state.agent_responses = None
        st.rerun()
    
    if st.session_state.learning_plan:
        if st.sidebar.button("📥 Export Plan"):
            plan_dict = asdict(st.session_state.learning_plan)
            json_str = json.dumps(plan_dict, indent=2, default=str)
            st.sidebar.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"learning_plan_{st.session_state.user_profile.name.lower().replace(' ', '_')}.json",
                mime="application/json"
            )
    
    # Tips
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Learning Tips")
    tips = [
        "Practice a little every day rather than cramming",
        "Immerse yourself in the language through media",
        "Don't be afraid to make mistakes - they're part of learning",
        "Find a language exchange partner",
        "Set small, achievable daily goals"
    ]
    
    for tip in tips:
        st.sidebar.markdown(f"• {tip}")

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="AI Language Learning Planner",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main header and description
    st.title("🗣️ AI Language Learning Planner")
    st.markdown("""
    Welcome to your personalized language learning journey! Our AI multi-agent system will analyze 
    your goals, create a custom curriculum, provide motivation strategies, and curate the best resources for you.
    """)
    
    # Main application logic
    if st.session_state.user_profile is None:
        # Show profile form
        profile = render_profile_form()
        
        if profile:
            st.session_state.user_profile = profile
            
            # Create learning plan using multi-agent system
            with st.spinner("🤖 Our AI agents are creating your personalized learning plan..."):
                coordinator = LanguageLearningCoordinator()
                
                try:
                    # This would normally be async, but Streamlit doesn't handle async well
                    # In a production app, you might want to use a different approach
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    learning_plan, agent_responses = loop.run_until_complete(
                        coordinator.create_comprehensive_plan(profile)
                    )
                    
                    st.session_state.learning_plan = learning_plan
                    st.session_state.agent_responses = agent_responses
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating learning plan: {str(e)}")
                    st.info("Please try again or check your OpenAI API configuration.")
    
    else:
        # Show learning plan
        if st.session_state.learning_plan and st.session_state.agent_responses:
            render_learning_plan(st.session_state.learning_plan, st.session_state.agent_responses)
        else:
            st.error("Learning plan not found. Please create a new plan.")

if __name__ == "__main__":
    main()
