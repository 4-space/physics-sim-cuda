#include <cuda.h>
#include "Objects.h"

//normalizes mass position
__global__ void normalizeMassPos(Mass* Masses, float* massPoints, float fieldDepth){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	switch(i%3):
		case (0):
		massPoints[i] = Masses[i/3].Position.x / fieldDepth;
		break;
		case (1):
		massPoints[i] = Masses[i/3].Position.y / fieldDepth;
		break;
		case (2):
		massPoints[i] = Masses[i/3].Position.z / fieldDepth;
		break;
}

//normalizes spring position
__global__ void normalizeSprings(Spring* Springs, float* springPoints, float fieldDepth){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	springPoints[i] = Springs[ i / 6].m1->Position.x / fieldDepth;
}

//calcuates force for each spring
__global__ void calcSpringForce(Spring* Springs){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float F = s.springConst * (Springs[i].initLen - Springs[i].springLen());
	
	//get direction of force
	glm::vec3 springForce = F *(glm::normalize(Springs[i].m1->Position - Springs[i].m2->Position)); //force in a direction
	
	//adding force to m1
	Springs[i].m1->accel += springForce * (1/(Springs[i].m1->mass));
	Springs[i].m2->accel += (-1.f * springForce) * (1/(Springs[i].m2->mass));
}

__global__ void gravityForce(Mass* m, int N, float gravConst, float dt){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(m[i].Posity.y <= -4.5){
		m[i].accel.y += -500 * (m[i].Position.y);
		m[i].Position.y = -4.5f;
	}else{m[i].accel.y += gravity;}
	
	m.updateVelocity(dt);
	m.updatePosition(dt);
}

__global__ void initMassAccel(Mass* m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	m[i].accel.x = 0.0f;
	m[i].accel.y = 0.0f;
	m[i].accel.z = 0.0f; 
};
