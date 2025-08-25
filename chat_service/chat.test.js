const request = require('supertest');
const app = require('./index');

describe('Chat Service', () => {
  it('health ok', async () => {
    const server = app.listen(0);
    const agent = request(server);
    const res = await agent.get('/health');
    expect(res.statusCode).toBe(200);
    server.close();
  });
});


