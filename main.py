from wtae.scenario import generate_multiagent_scenario
import numpy as np

def main():
    # Generate the multi-agent scenario
    scenario = generate_multiagent_scenario()

    # Extract the world and callbacks
    world = scenario["world"]
    reset_callback = scenario["reset_callback"]
    reward_callback = scenario["reward_callback"]
    observation_callback = scenario["observation_callback"]

    # Reset the world
    reset_callback()

    done = False
    while not done:
        world.render()

        # Randomly assign actions for each agent
        actions = np.random.choice(world.num_targets + 1, world.num_weapons)

        # Use callbacks for reward and observation
        reward = reward_callback(actions)
        observations = observation_callback()

        print(f"Actions: {actions}")
        print(f"Reward: {reward}")
        print(f"Observations: {observations}")

        # Check if all targets are removed
        done = world.done

    print("All targets have been neutralized or escaped!")


if __name__ == "__main__":
    main()
