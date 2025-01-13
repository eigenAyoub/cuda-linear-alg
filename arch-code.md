
int image_id = 27;
std::cout << y_train[27] << "\n\n";
for (int i=0; i < IMAGE_WIDTH; i++){
    for (int j=0; j < IMAGE_WIDTH; j++){
        int pix = X_train[image_id*IMAGE_SIZE + i*IMAGE_WIDTH + j];
        std::cout << (!pix ? "#" : " ");
    }
    std::cout << "\n";
}