#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <unistd.h>
#include <vector>
#include "Objects.h"
#include <stdio.h>

/*Defined Constatns*/
#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
using namespace std;


/*Global Variables*/
bool step = false;
GLfloat angle = 0;
GLfloat angle2 = 0;
GLfloat movY = 0;
const float gravity = -9.81f;
const float dt = .005f;
const float fieldDepth = 5.f; //in meters
vector<Mass> massList;
vector<Spring> springList;
int springListSize;
int massListSize;

//These are initialized and passed to the GPU
Mass* d_masslist;
Spring* d_springList;

//this makes it easy to copy data back and forth! between device and host!
Mass* h_massList;
Spring* h_springList;

/*GLfloat vertices[] =	//mass cooridnates in a cube
    {
        -1, -1, -1,   -1, -1,  1,   -1,  1,  1,   -1,  1, -1,
        1, -1, -1,    1, -1,  1,    1,  1,  1,    1,  1, -1,
        -1, -1, -1,   -1, -1,  1,    1, -1,  1,    1, -1, -1,
        -1,  1, -1,   -1,  1,  1,    1,  1,  1,    1,  1, -1,
        -1, -1, -1,   -1,  1, -1,    1,  1, -1,    1, -1, -1,
        -1, -1,  1,   -1,  1,  1,    1,  1,  1,    1, -1,  1
    };*/

GLfloat vertices[] =	//mass cooridnates in a cube
    {
        -1,1,1,   1, 1,1,  -1,-1,1,  1,-1,1,
        -1,1,-1,  1, 1,-1, -1,-1,-1, 1,-1,-1,
        -3,1,1,  -3,-1,1,  -3,1,-1, -3,-1,-1,
         3,1,1,   3,-1,1,   3,1,-1,  3,-1,-1};

GLfloat colors[] =
	{
		0,1,0, 0,1,0, 0,1,0, 0,1,0,
		0,1,0, 0,1,0, 0,1,0, 0,1,0,
		0,1,0, 0,1,0, 0,1,0, 0,1,0,
		0,1,0, 0,1,0, 0,1,0, 0,1,0,
	};

GLfloat plane_color [] = 
{255, 102, 153,  255, 77, 77,  255,102, 153,  255, 77, 77};

GLfloat groundp[] = 
{ -5, 0, 5,   5, 0, 5,   5, 0, -5,   -5, 0, -5};

/*Error Handling Functions*/
void die(const char *err){
	perror(err);
	exit(1);
}

/*Control Function*/
void controls(GLFWwindow* window, int key, int scancode, int action, int mods){
	if(action == GLFW_PRESS){
        if(key == GLFW_KEY_ESCAPE){
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        else if(key == GLFW_KEY_N){
        	if(step){step = false;}
        	else{step = true;} 
        }
        else if(key == GLFW_KEY_UP){
            angle += 2.f;
        }else if(key == GLFW_KEY_DOWN){
        	angle -= 2.f;
        }
         else if(key == GLFW_KEY_LEFT){
         	angle2 += 2.f;
         }
         else if(key == GLFW_KEY_RIGHT){
         	angle2 -= 2.f;
         }
         else if(key == GLFW_KEY_W){
         	movY += .1f;
         }
         else if(key == GLFW_KEY_S){
         	movY -= .1f;
         }
    }
}


void initList(){
	//initialize objects in scene
	
	//There are 8 massess in this cube system
	/*
	int n = 16;

	for(int i = 0; i < n; i++){
		massList.push_back(Mass(.5));
	}
	*/
	/*
	int[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; 
	*/

	for (int i = 0; i < 16; i++){
		massList.push_back(Mass(.5));
	}

	massListSize = massList.size();
	
	int i = 0;
	for(Mass& m: massList){
		m.setPosition(vertices[i], vertices[i+1], vertices[i+2]);
		i += 3;
	}
	
	/*There are n springs in this system*/
	

	for(i = 0; i < massListSize; i++){
		for(j = i+1; j < massListSize; j++){
			springList.push_back(Spring(&massList[i], &masslist[j], 200.3f));
		}
	}

	/*
	int indx1[] = {0,2,4,6,9,10,11};
	int indx2[] = {0,2,4,6,10,11};
	int indx3{} = {0,2,4,6,11};
	int indx4[] = {0,2,4,6};
	int indx5[] = {1,2,5,7,13,14,15};
	int indx6[] = {1,3,5,7,14,15};
	int indx7[] = {1,3,5,7,15};
	int indx8[] = {1,3,5,7}; 
	*/

	//this is unfortunate but necessary because of the way that I initialized the mass positions :(
	springList.push_back(Spring(&massList[8], &massList[0], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[2], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[4], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[6], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[9], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[10], 200.3f));
	springList.push_back(Spring(&massList[8], &massList[11], 200.3f));
	
	springList.push_back(Spring(&massList[9], &massList[0], 200.3f));
	springList.push_back(Spring(&massList[9], &massList[2], 200.3f));
	springList.push_back(Spring(&massList[9], &massList[4], 200.3f));
	springList.push_back(Spring(&massList[9], &massList[6], 200.3f));
	springList.push_back(Spring(&massList[9], &massList[10], 200.3f));
	springList.push_back(Spring(&massList[9], &massList[11], 200.3f));
	
	springList.push_back(Spring(&massList[10], &massList[0], 200.3f));
	springList.push_back(Spring(&massList[10], &massList[2], 200.3f));
	springList.push_back(Spring(&massList[10], &massList[4], 200.3f));
	springList.push_back(Spring(&massList[10], &massList[6], 200.3f));
	springList.push_back(Spring(&massList[10], &massList[11], 200.3f));
	
	springList.push_back(Spring(&massList[11], &massList[0], 200.3f));
	springList.push_back(Spring(&massList[11], &massList[2], 200.3f));
	springList.push_back(Spring(&massList[11], &massList[4], 200.3f));
	springList.push_back(Spring(&massList[11], &massList[6], 200.3f));

	springList.push_back(Spring(&massList[12], &massList[1], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[3], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[5], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[7], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[13], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[14], 200.3f));
	springList.push_back(Spring(&massList[12], &massList[15], 200.3f));
	
	springList.push_back(Spring(&massList[13], &massList[1], 200.3f));
	springList.push_back(Spring(&massList[13], &massList[3], 200.3f));
	springList.push_back(Spring(&massList[13], &massList[5], 200.3f));
	springList.push_back(Spring(&massList[13], &massList[7], 200.3f));
	springList.push_back(Spring(&massList[13], &massList[14], 200.3f));
	springList.push_back(Spring(&massList[13], &massList[15], 200.3f));
	
	springList.push_back(Spring(&massList[14], &massList[1], 200.3f));
	springList.push_back(Spring(&massList[14], &massList[3], 200.3f));
	springList.push_back(Spring(&massList[14], &massList[5], 200.3f));
	springList.push_back(Spring(&massList[14], &massList[7], 200.3f));
	springList.push_back(Spring(&massList[14], &massList[15], 200.3f));

	springList.push_back(Spring(&massList[15], &massList[1], 200.3f));
	springList.push_back(Spring(&massList[15], &massList[3], 200.3f));
	springList.push_back(Spring(&massList[15], &massList[5], 200.3f));
	springList.push_back(Spring(&massList[15], &massList[7], 200.3f));
	
	springListSize = springList.size() * 2;

	//Turn std::vector objects into pointer arrays because CUDA doesn't work with std :(
	h_massList = (Mass*) malloc(massListSize * sizeof(Mass));
	for(i = 0; i < massListSize; i++){
		h_massList[i] = &massList[i];
	}

	h_springList = (Spring*) malloc(springListSize/2 * sizeof(Spring));
	for(i = 0; i < springListSize/2; i++){
		h_springList[i] = springList[i]; //shallow copy as needed
	}
}


/*Drawing Functions*/
GLFWwindow* initWindow(const int x, const int y){ //initalize the window that application is in	
	if(!glfwInit())
		die("GLFWinit failed.");
	GLFWwindow* window = glfwCreateWindow(x, y, "Physics Sim", NULL, NULL);
	if(window == NULL){
		glfwTerminate();
		die("Could not create window.");
	}

	glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, controls); 
    for(int i = 0; i < (sizeof(plane_color)/ sizeof(plane_color[0])); i++){
    	plane_color[i] /= 255;
    }
    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_BACK);
    glPointSize(10);
	return window;
}

/*Force Adding Functions*/

	//springs are in between masses. 

	//1 calculate all the forces acting on each mass
	//# of forces = #of springs + gravity

void drawCube(){
	float massPoints[massList.size() * 3];
	float springPoints[springList.size() * 6];
	
	

	//while the simulation is running
	/*
	for(Mass& m : massList){
		m.accel.x = 0.0f;
		m.accel.y = 0.0f;
		m.accel.z = 0.0f;
	}*/

	//cudaMalloc...
	//cudaMemcpy

	cudaMalloc( (void**) d_masslist, massListSize*sizeof(Mass));
	cudaMemcpy(d_masslist, h_massList, massListSize*sizeof(Mass) , cudaMemcpyHostToDevice);
	
  //run initMassAccel with 3 threads per block and massList.size() blocks
	initMassAccel<<<massList.size(), 3>>>(d_masslist);
	cudaMemcpy(d_masslist, h_massList, massListSize*sizeof(Mass), cudaMemcpyDeviceToHost );
	
	/*
	for(Spring& s : springList){
		float F = s.springConst * (s.initLen - s.springLen());

		//get direction of force
		//Position pos2;
		glm::vec3 springForce = F *(glm::normalize(s.m1->Position - s.m2->Position)); //force in a direction
		
		//adding force to m1
		s.m1->accel += springForce * (1/(s.m1->mass));
		s.m2->accel += (-1.f * springForce) * (1/(s.m2->mass));
	} */

	//cudaMalloc...
	//cudaMemcpy
	
	cudaMalloc((void**) d_springList, springList.size()*sizeof(Spring));
	cudaMemcpy(d_springList, h_springList, springListSize*sizeof(Mass), cudaMemcpyHostToDevice);

  //<<num strings, one thread per spring>>
	calcSpringForce<<<springList.size(), 1 >>>(d_springList);

	cudaMemcpy(h_springList, d_springList, springList.size()*sizeof(Spring), cudaMemcpyDeviceToHost);

	/*Update Gravity and Ground Normal Force*/
	
	/*
	for(Mass& m : massList){
		if(m.Position.y <= -4.5){
			m.accel.y += -500 * (m.Position.y);
			m.Position.y = -4.5f;
		}else{
		m.accel.y += gravity;}
		m.updateVelocity(dt);
		m.updatePosition(dt);
	}*/

	//cudaMalloc...
	//cudaMemcpy
	

	gravityForce<<<massList.size(), 1 >>>(d_masslist, Gravity, dt);

	/*Get a list of points that represent the new positions of the masses*/
	/*
	int i = 0; //fieldDepth is a value that puts the values from -500 and 500 between -1 and 1
	for(Mass& m: massList){
		massPoints[i] = m.Position.x / fieldDepth;
		massPoints[i+1] = m.Position.y / fieldDepth;
		massPoints[i+2] = m.Position.z / fieldDepth;
		i+=3;
	}*/

	//cudaMalloc...
	//cudaMemcpy
	normalizeMassPos<<<massList.size(), 3>>>(d_masslist, d_mass_points, fieldDepth);

	/*Get a list of points that represent the positions of the spring end points*/
	
	/*
	i = 0;
	for(Spring& s : springList){
		springPoints[i] =  s.m1->Position.x / fieldDepth;
		springPoints[i+1] = s.m1->Position.y / fieldDepth;
		springPoints[i+2] = s.m1->Position.z / fieldDepth;
		springPoints[i+3] = s.m2->Position.x / fieldDepth;
		springPoints[i+4] = s.m2->Position.y / fieldDepth;
		springPoints[i+5] = s.m2->Position.z / fieldDepth;
		i += 6;
	}
	*/
	
	//cudaMalloc...
	//cudaMemcpy
	normalizeSprings<<<springListSize, 6>>>(d_springList, d_springPoints, fieldDepth);

	cudaMemcpy(springList, d_springList, springList.size()*sizeof(Spring), cudaMemcpyDeviceToHost);
	cudaMemcpy(springPoints, d_springPoints, springList.size()*6*sizeof(GL_FLOAT), cudaMemcpyDeviceToHost);

	/*Reset Modelview_Matrix + Adjustments*/
	glMatrixMode(GL_MODELVIEW_MATRIX);
	glPushMatrix();
	glTranslatef(0.f, movY, 0.f);
	/*Enable Drawing*/
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY); 
	/*Draw Vertices*/
	glVertexPointer(3, GL_FLOAT, 0, massPoints);
	glColorPointer(3, GL_FLOAT, 0, colors);
	glDrawArrays(GL_POINTS, 0, massSize);
	glDisableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, springPoints);
	glDrawArrays(GL_LINES, 0, springListSize);
	/*Disable Drawing*/
	
	glDisableClientState(GL_VERTEX_ARRAY);
	//glLoadIdentity();
	glPopMatrix();		
}

void drawGround(){ //draw the ground the cube starts above	
	/*Enable Drawing*/
	//glMatrixMode(GL_PROJECTION_MATRIX);
	glPushMatrix();
	glTranslatef(0.f, -0.9f, 0.f);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY); 
	glColorPointer(3, GL_FLOAT, 0, plane_color);
	glVertexPointer(3, GL_FLOAT, 0, groundp);
	glDrawArrays(GL_QUADS, 0, 4);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glPopMatrix();
}

void display(GLFWwindow* window){ // the main drawing function
	
	while(!glfwWindowShouldClose(window)){
		if(step){		
		/*Clear Screen*/
		glClearColor(0.0, 0.1, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/*Reset Projection_Matrix*/
		glMatrixMode(GL_PROJECTION_MATRIX);
		glLoadIdentity();


		//glTranslatef(); //use to translate structure
		
		glRotatef(angle, 1.f, 0.f, 0.f); //rotate the world!
		glRotatef(angle2, 0.f, 1.f, 0.f); //rotate the world!
		drawCube();
		drawGround();
		
		
		/*Display Updates to Screen*/
		glfwSwapBuffers(window);
		}
		glfwPollEvents();
	}
}

int main(int argc, char **argv){
	initList();
	GLFWwindow* window = initWindow(SCREEN_WIDTH, SCREEN_HEIGHT);
	display(window);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
//calculate the velocity of each mass
		//calculate the position for each mass
		//update the position each mass
			//NOTE: if position is below ground 
					//position is 0, acceleration is F = k(0-d)



//updateAccel() must be run first so that the mass has correct value of acceration for the time period



