// Kernel definition
__global__ tpsCuda VecAdd(float*** imageCoord, float* solutionX, float* solutionY, float* refKeyX, float* refKeyY, int numOfKeys)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    float newX = solutionX[0] + x*solutionX[1] + y*solutionX[2];
	float newY = solutionY[0] + x*solutionY[1] + y*solutionY[2];

	for (uint i = 0; i < referenceKeypoints_.size(); i++) {
		float r = (x-xi)*(x-xi) + (y-yi)*(y-yi);
		if (r != 0.0) {
			newX += r*log(r) * solutionX[i+3];
			newY += r*log(r) * solutionY[i+3];
		}
	}
	imageCoord[x][y][0] = newX;
	imageCoord[x][y][1] = newY;
}