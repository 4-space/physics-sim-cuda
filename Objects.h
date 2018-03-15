#include <glm/glm.hpp>

/*Struct Definitions*/
struct Mass{
	float mass;
	glm::vec3 vel;
	glm::vec3 accel;
	glm::vec3 Position;
	
	void updatePosition(float dt){
	Position += vel * dt;
	}
	void updateVelocity(float dt){
	vel += accel * dt;
	}
	void setPosition(float x, float y, float z){
		Position.x = x;
		Position.y = y;
		Position.z = z; 
	}
	/*Constructor*/
	Mass(float m, float x = 0, float y = 0, float z = 0) 
	:mass(m), Position(glm::vec3(z, y, z)), accel(glm::vec3(0)), vel(glm::vec3(0))
	{}	
};

/*Weird wrapper thing for mass distance*/
float getDistance(Mass &m1, Mass &m2){
	return glm::distance(m1.Position, m2.Position);
}

struct Spring{
	float springConst;
	float initLen;
	Mass *m1;
	Mass *m2;
	
	/*Returns Spring Length*/
	float springLen(){
		return getDistance(*m1, *m2);
	}
	
	/*Constructor*/
	Spring(Mass *n1, Mass *n2, float k = 1.f)
	:m1(n1), m2(n2), springConst(k)
	{
		initLen = getDistance(*n1, *n2);
	}
};
/*End Struct Definitions*/
