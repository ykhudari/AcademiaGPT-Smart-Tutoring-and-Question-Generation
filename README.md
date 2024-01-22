# AcademiaGPT: Smart Tutoring and Question Generation 

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="python" width="120" height="30"/> </a>  
<a href="https://chat.openai.com/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT" width="120" height="30"/> </a> 
<a href="https://jupyter.org/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="jupyter" width="120" height="30"/> </a>
<a href="https://supabase.com/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Supabase-181818?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase" width="120" height="30"/> </a>
<a href="https://fastapi.tiangolo.com/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white" alt="FastAPI" width="120" height="30"/> </a>
<a href="https://www.atlassian.com/software/jira" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=Jira&logoColor=white" alt="Jira" width="120" height="30"/> </a>
<a href="https://www.notion.so/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white" alt="Notion" width="120" height="30"/> </a>
<a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" width="120" height="30"/> </a>
</p>  

### Project Demonstration - https://www.linkedin.com/feed/update/urn:li:activity:7149602832260689920/  

### Project Information - https://www.youtube.com/watch?v=E_6OqPSRwxg  

# Abstract

Artificial Intelligence (AI) in education is utilized to enhance the learning and teaching experiences for both students and instructors. These technologies have helped students and
educators across the world in promoting inclusivity in education. They not only have the potential to make education more accessible to diverse learners, but they also make the learning process more interactive and enjoyable.Our project aims at mobilizing this exciting development in AI to build a learning tool for the Jupyteach platform through Large Language Models, utilizing Retrieval Augmented Generation (RAG). There are mainly two modules associated with this platform, the AI Tutor and Question Generator.

#### AI Tutor: 
The purpose of the AI Tutor is to help give students a personalized learning
experience of specific topics through interactive methods. It also aims at engaging users in a dynamic and responsive learning environment in acquiring new skills and gaining knowledge.

#### Question Generator: 
The purpose of the question generator is to assist the instructors in creating questions for educational and assessment purposes.

# Project Overview

**Objective**: Provide automated, personalized explanations for student questions.
**Goal**: Reduce confusion, enhance understanding, and improve learning outcomes.
**Minimum Viable Product Focus**: Core natural language processing capabilities for a single subject area.

### Minimum Viable Product (MVP):

**Functionality**: Students ask text-based questions for automated, personalized explanations.
**Emphasis**: Written responses in layman's terms for viability.

### User Stories:

**Student**: Ask conceptual questions for explanations on foundational topics.
**Instructor**: Use question generator for quiz and exam design.

### Additional Explanation Formats:

1. Code Examples:

- Display code snippets to illustrate concepts.
- Especially beneficial for technical/programming material.

2. Diagrams:

- Visual representation of relationships between concepts.
- Aids visual learners in understanding.

3. Concept Workflow:

- Mapping logic flows around decision points.
- Aids understanding of complex procedures.

### AI Tutor Capabilities:

- Expandable capabilities over time.
- Personality and tone variations: Formal vs informal, strict vs friendly, serious vs conversational.
- Feedback gathering: Through usage, interviews, and surveys.

### Iterative Development Process:

**Starting Narrow**: Analyze logs of questions, explanations, feedback, and trouble areas.  
**Expansion**: Inform expansion to new topics based on insights. Continuous adjustment as the knowledge base grows.

### Monitoring Metrics:

- Explanation quality and question similarity matches.
- Essential for ensuring adequate and reliable coverage.
- Facilitates handling increased question complexity and variety over time.

# High-level Architecture Diagram

<img width="468" alt="image" src="https://github.com/jainammshahh/AcademiaGPT-Smart-Tutoring-and-Question-Generation/assets/114266749/76a605f5-8765-43a7-be41-cc0769c92e12">

The high level diagram shows the following overview of AcademiaGPT:

- Conceptual overview of project development steps.
- Major components/modules identification.
- Description of interactions and dependencies.
- Early identification of risks and challenges.
- Foundation for detailed design and implementation.
- Scalability and flexibility for future changes.

### Components of the System:

1. User-Friendly Interface:

- Web application for students and tutors.
- Access to lessons and teaching material generation.

2. Server and Interface:

- Manages user requests and usage data.
- Records topics explored and learning history for personalized experiences.

3. Adaptive Learning Engine:

- Designed based on RAG.
- Extracts key concepts and identifies user intent.

4. Knowledge Base:

- Database storing vectorized data.
- Domain-specific information for AI tutor or question generator.
- Constantly updated with new content.

5. Student Module:

- Maintains user's current knowledge, skills, and learning style.
- Recommends learning materials based on user profile.

6. Pedagogical Module:

- Determines teaching strategies and question difficulty.
- Highly adaptive to different dialects and informal language.

7. Answer Generation Module:

- Retrieves information from knowledge base.
- Formulates summarized and informative responses.
- Generates various types of answers (explanations, examples, demonstrations)

# Critical Path

<img width="700" alt="image" src="https://github.com/jainammshahh/AcademiaGPT-Smart-Tutoring-and-Question-Generation/assets/114266749/61afb14a-4ff0-4a2d-8cba-04d9da5f5642">

# Final Contemplation

In navigating the development of our MVP, we faced challenges with the Question Generator's design constraints, specifically the mandatory use of the Pydantic Output model. This posed limitations on prompt engineering.

Looking forward, we contemplated towards enhancement of the Question Generator's interactivity and adopting a more conversational tone. Increasing the difficulty of user-generated questions was a key focus for an enriched learning experience. Additionally, addressing AI Tutor challenges, such as hallucination during output retrieval, also proved crucial for refining its performance.

Our immediate roadmap involves creating a user-friendly interface for the Question Generator, prioritizing user experience, and implementing a robust data protection strategy. Simultaneously, the AI Tutor will undergo refinement based on real-world usage insights.

In our academic journey, business development exposed us to state-of-the-art tools and technologies like Generative AI, Prompt Engineering, and Retrieval Augmented Generation, enriching our software development understanding.


