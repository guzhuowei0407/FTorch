program transformer_inference
   
   use, intrinsic :: iso_fortran_env, only : wp => real32

   ! Import Ftorch library
   use ftorch, only : torch_model, torch_tensor, torch_tensor_from_array, &
                      torch_tensor_to_array, torch_model_load, torch_model_forward, &
                      torch_delete, torch_kCPU

   implicit none

   call main()

contains

   subroutine main()

      integer :: i

      ! Set up types of input and output data
      type(torch_model) :: model
      type(torch_tensor), dimension(1) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors
      

      real(wp), dimension(5,2), target :: in_data = reshape([ &
        0.0_wp, 0.0_wp, &
        45.0_wp, 90.0_wp, &
        -45.0_wp, -90.0_wp, &
        90.0_wp, 180.0_wp, &
        -90.0_wp, -180.0_wp], [5,2])

      real(wp), dimension(5,1) :: out_data

      ! Layout definitions for tensors
      integer, parameter :: in_layout(2) = [1, 2]
      integer, parameter :: out_layout(2) = [1, 2]

      ! Call pre-trained transformer model, torchscript
      call torch_model_load(model, "../transformer_precipitation_model.pt")
      
      ! Map Fortran arrays to PyTorch tensors
      call torch_tensor_from_array(in_tensors(1), in_data, in_layout, torch_kCPU)
      call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)
        
      ! Print results
      write(*, '(A)') "Input Data (Latitude, Longitude):"
      do i = 1, size(in_data, 1)
         write(*, '(F10.2, F10.2)') in_data(i, 1), in_data(i, 2)
      end do

      ! Print predicted precipitation
      write(*, '(A)') "Predicted Precipitation:"
      do i = 1, size(out_data, 1)
         write(*, '(F10.4)') out_data(i, 1)
      end do

      ! Clean up resources
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)

   end subroutine main

end program transformer_inference
