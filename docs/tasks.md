# Supported Tasks

## T1 Door Wedge & Pass-Through

Planner intent:

- identify a valid wedgeable door
- choose a plausible holder and passer
- stage one robot to hold while the partner crosses
- release and follow after the partner clears the doorway

Terminal semantics:

- both robots must reach the target-side region

## T2 Herding / Corralling

Planner intent:

- identify a movable target object and a goal region
- place one robot behind the object as pusher
- place the partner on the goal-side funnel role
- push while constraining the escape angle

Terminal semantics:

- the object must enter the goal region

## T4 Collaborative Search & Converge

Planner intent:

- partition search sectors between the robots
- allow one robot to find and report the target
- bring the partner to the target location after discovery

Terminal semantics:

- both robots converge to the target region

## T6 Relay Delivery

Planner intent:

- identify a movable payload and a far goal
- choose a plausible handoff region
- let the starter push to handoff
- let the finisher take over and deliver to goal

Terminal semantics:

- the payload must reach the goal region
