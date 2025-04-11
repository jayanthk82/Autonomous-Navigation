# 🤖 Autonomous Navigation Simulation Based on Semantic Commands
🧠 A simulation-based project that demonstrates how a robot could understand semantic commands (e.g., “go to kitchen”) and learn to navigate through a 2D environment using Reinforcement Learning and symbolic room structures.

⚠️ Note: This is a simulated sampling project. It does not implement real-world audio processing or robotic control but lays the foundation for those features.

📌 Project Overview
This project simulates an autonomous navigation strategy for a humanoid robot that navigates in a 2D grid environment with:

Named rooms

Walls and doors

Obstacles

It includes:

🎯 Semantic command interpretation (simulated input like "go to kitchen")

🧠 Path planning using Q-learning

🧱 Visualization of the path across doors and around obstacles

📦 The goal is to test intelligent path sampling and environment awareness in a controlled grid environment before integrating real-world robotics.

🔍 What This Project Does:
✅ Simulates a multi-room environment
✅ Allows room-to-room navigation via doors
✅ Trains a Q-learning agent to find valid paths
✅ Accepts textual semantic commands (simulated audio)
✅ Provides visual feedback of navigation paths

🚫 What This Project Does Not (Yet):
❌ Real-time audio command recognition
❌ Sound source localization using microphones
❌ Hardware control of an actual robot
❌ Real-world SLAM or depth sensing

These will be part of future versions built on top of this simulation layer.

🔗 Combining the Data (Q-values + Environment)
All Q-values are stored and can be reused to simulate navigation for different commands.

The agent combines:

Room labels (semantic meaning)

Q-table values (learned paths)

Map structure (walls, obstacles, doors)

This shows how semantic goals can be mapped to physical actions in a basic AI agent.

💡 Future Work
Integrate real-time audio command parsing

Implement microphone array-based sound localization

Deploy on physical humanoid robot (e.g., NANO)

Use DQN or PPO for more robust policy learning

Add dynamic environments (moving obstacles)

👨‍💻 Author
Made with ❤️ by an aspiring AI engineer exploring real-world robotics via simulation.

📫 your-JAYANTH @ jayanthkonanki82@gmail.com
