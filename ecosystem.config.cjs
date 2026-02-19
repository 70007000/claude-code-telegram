module.exports = {
  apps: [{
    name: 'claude-telegram',
    script: '.venv/bin/python',
    args: '-m src.main',
    cwd: '/home/fer/claude-code-telegram',
    autorestart: true,
    max_restarts: 5,
    restart_delay: 5000,
  }]
};
