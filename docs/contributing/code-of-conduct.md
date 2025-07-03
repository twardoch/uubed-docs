# Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

- **Being Respectful**: Using welcoming and inclusive language
- **Being Collaborative**: Accepting constructive criticism gracefully
- **Being Supportive**: Focusing on what is best for the community
- **Being Empathetic**: Showing empathy towards other community members
- **Being Professional**: Maintaining professional conduct in all interactions

Examples of unacceptable behavior by participants include:

- **Harassment**: The use of sexualized language or imagery and unwelcome sexual attention or advances
- **Trolling**: Insulting/derogatory comments, and personal or political attacks
- **Privacy Violations**: Publishing others' private information without explicit permission
- **Misconduct**: Other conduct which could reasonably be considered inappropriate in a professional setting

## Technical Community Standards

### Code Review Ethics

- **Focus on Code, Not People**: Critique the code and ideas, not the person
- **Be Constructive**: Provide specific, actionable feedback
- **Be Patient**: Remember that everyone is learning and improving
- **Be Thorough**: Take time to understand the context before commenting

### Issue and Discussion Guidelines

- **Search First**: Check if your question or issue has been addressed before
- **Be Clear**: Provide enough context for others to understand and help
- **Stay On Topic**: Keep discussions relevant to the project
- **Be Patient**: Maintainers and contributors are often volunteers

### Examples of Good Technical Communication

#### Constructive Code Review

```markdown
# Good ✅
"This function could be more efficient. Consider using a dictionary lookup 
instead of iterating through the list. Here's an example: [code snippet]"

# Bad ❌  
"This is slow and wrong."
```

#### Helpful Issue Reporting

```markdown
# Good ✅
"I'm experiencing a validation error when encoding embeddings with negative 
values. Steps to reproduce: [detailed steps]. Environment: Python 3.11, 
uubed 1.0.2. Error message: [full traceback]"

# Bad ❌
"It doesn't work."
```

#### Supportive Discussion

```markdown
# Good ✅
"Welcome to the project! That's an interesting use case. Have you considered 
approach X? Here are some resources that might help: [links]"

# Bad ❌
"This has been asked before. RTFM."
```

## Our Responsibilities

### Maintainers

Project maintainers are responsible for:

- **Clarifying Standards**: Making sure community standards are clear and understood
- **Fair Enforcement**: Enforcing the code of conduct consistently and fairly
- **Corrective Action**: Taking appropriate action in response to unacceptable behavior
- **Leadership**: Setting a positive example through their own behavior

### Contributors

All contributors are expected to:

- **Follow Guidelines**: Adhere to technical and social guidelines
- **Report Issues**: Report violations of the code of conduct to maintainers
- **Support Others**: Help create a welcoming environment for new contributors
- **Learn Continuously**: Be open to feedback and willing to improve

## Scope

This Code of Conduct applies to all project spaces, including:

- **GitHub Repositories**: Issues, pull requests, discussions, and code
- **Documentation**: All forms of project documentation
- **Communication Channels**: Any official communication channels
- **Events**: Project-related conferences, meetups, or presentations
- **Social Media**: When representing the project on social platforms

## Enforcement

### Reporting Violations

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by:

1. **GitHub Issues**: For public violations, you may comment on the relevant issue/PR
2. **Private Contact**: Email project maintainers directly at [contact email]
3. **GitHub Reporting**: Use GitHub's built-in reporting mechanisms

### Investigation Process

All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident.

### Enforcement Guidelines

Project maintainers will follow these Community Impact Guidelines in determining the consequences for any action they deem in violation of this Code of Conduct:

#### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from project maintainers, providing clarity around the nature of the violation and an explanation of why the behavior was inappropriate. A public apology may be requested.

#### 2. Warning

**Community Impact**: A violation through a single incident or series of actions.

**Consequence**: A warning with consequences for continued behavior. No interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, for a specified period of time. This includes avoiding interactions in community spaces as well as external channels like social media. Violating these terms may lead to a temporary or permanent ban.

#### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public communication with the community for a specified period of time. No public or private interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, is allowed during this period. Violating these terms may lead to a permanent ban.

#### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the project community.

## Technical Conflict Resolution

### Disagreements on Technical Decisions

When technical disagreements arise:

1. **Discussion**: Start with respectful discussion in issues or pull requests
2. **Documentation**: Document different approaches and trade-offs
3. **Consensus Building**: Seek input from multiple community members
4. **Maintainer Decision**: Maintainers make final decisions when consensus isn't reached
5. **Appeal Process**: Decisions can be appealed through private communication

### Example: Handling Technical Disagreements

```markdown
# Productive Technical Discussion
Person A: "I think we should use approach X because it's more performant."
Person B: "I see the performance benefit, but approach Y is more maintainable. 
          Here's a benchmark showing the trade-offs: [data]"
Person C: "Both approaches have merit. Could we implement Y now for 
          maintainability and optimize with X later if needed?"
Maintainer: "Great discussion. Let's go with Y for now as maintainability 
           is our current priority. We can revisit optimization in v2.0."
```

## Building an Inclusive Community

### Welcoming New Contributors

- **Mentorship**: Offer guidance to newcomers
- **Good First Issues**: Label issues appropriate for beginners
- **Documentation**: Maintain clear setup and contribution guides
- **Recognition**: Acknowledge all forms of contribution

### Supporting Diversity

We actively work to:

- **Remove Barriers**: Make contributing accessible to people with different backgrounds and experience levels
- **Inclusive Language**: Use language that welcomes all contributors
- **Multiple Perspectives**: Value diverse viewpoints and approaches
- **Equal Opportunity**: Ensure all contributors have equal opportunity to participate

### Language and Communication

We strive to use:

- **Inclusive Pronouns**: Use people's preferred pronouns
- **Clear Communication**: Avoid jargon when simpler language works
- **Patient Explanation**: Take time to explain concepts thoroughly
- **Cultural Sensitivity**: Be aware of cultural differences in communication styles

## Positive Community Examples

### Mentoring New Contributors

```markdown
"Hi @newcontributor! Welcome to the project. I see you're interested in 
working on the documentation. That's fantastic! 

To get started:
1. Check out our [setup guide](setup.md)
2. Look at the 'good first issue' labeled items
3. Feel free to ask questions in this issue or reach out directly

Don't hesitate to submit a draft PR - we're happy to provide feedback 
and help you learn our processes. Thanks for contributing!"
```

### Constructive Problem Solving

```markdown
"I notice this implementation might have performance implications with 
large datasets. Would you like to collaborate on optimizing it? I have 
some ideas we could explore together:

1. Batch processing approach
2. Memory-efficient streaming
3. Caching frequently accessed data

Happy to pair program or share some benchmark code to test different 
approaches. What do you think?"
```

## Questions and Clarifications

If you have questions about this Code of Conduct:

- **Read the Full Document**: Make sure you've read this entire document
- **Check Examples**: Look at the examples provided throughout
- **Ask Publicly**: Post questions in GitHub Discussions
- **Contact Maintainers**: Reach out privately if needed

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 2.0, available at https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

Community Impact Guidelines were inspired by [Mozilla's code of conduct enforcement ladder](https://github.com/mozilla/diversity).

For answers to common questions about this code of conduct, see the FAQ at https://www.contributor-covenant.org/faq. Translations are available at https://www.contributor-covenant.org/translations.

---

**Remember**: We're all here because we're passionate about building great software together. Let's keep it positive, productive, and fun! 🚀

## Related Topics

- [Contributing Guidelines](guidelines.md)
- [Development Setup](setup.md)