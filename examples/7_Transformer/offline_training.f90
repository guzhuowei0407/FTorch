program offline_training_inference

   use, intrinsic :: iso_fortran_env, only : wp => real32

   ! Import Ftorch library
   use ftorch, only : torch_model, torch_tensor, torch_tensor_from_array, &
                      torch_tensor_to_array, torch_model_load, torch_model_forward, &
                      torch_delete, torch_kCPU

   implicit none

   call main()

contains

   subroutine main()

      integer :: i, num_vars, num_samples, num_outputs

      ! Set up types of input and output data
      type(torch_model) :: model
      type(torch_tensor), dimension(1) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors

      ! Define constants
      integer, parameter :: num_vars = 72        ! Number of input variables
      integer, parameter :: num_samples = 3168  ! Number of input samples
      integer, parameter :: num_outputs = 7     ! Number of output variables
      integer, parameter :: output_samples = 634 ! Number of output samples

      ! Declare arrays for all 72 variables
      real(wp), dimension(num_samples) :: landfrac, Latitude, Longitude, FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT, LANDFRAC_PFT
      real(wp), dimension(num_samples) :: PCT_NATVEG, AREA, peatf, abm, SOIL_COLOR, SOIL_ORDER, SOIL3C, SOIL4C
      real(wp), dimension(num_samples) :: DEADSTEMC, DEADCROOTC, CWDC, TLAI, GPP
      real(wp), dimension(num_samples) :: PCT_NAT_PFT_0, PCT_NAT_PFT_1, PCT_NAT_PFT_2, PCT_NAT_PFT_3, PCT_NAT_PFT_4, PCT_NAT_PFT_5
      real(wp), dimension(num_samples) :: PCT_NAT_PFT_6, PCT_NAT_PFT_7, PCT_NAT_PFT_8, PCT_NAT_PFT_9, PCT_NAT_PFT_10
      real(wp), dimension(num_samples) :: PCT_NAT_PFT_11, PCT_NAT_PFT_12, PCT_NAT_PFT_13, PCT_NAT_PFT_14, PCT_NAT_PFT_15, PCT_NAT_PFT_16
      real(wp), dimension(num_samples) :: PCT_SAND_0, PCT_SAND_1, PCT_SAND_2, PCT_SAND_3, PCT_SAND_4, PCT_SAND_5
      real(wp), dimension(num_samples) :: PCT_SAND_6, PCT_SAND_7, PCT_SAND_8, PCT_SAND_9
      real(wp), dimension(num_samples) :: SCALARAVG_vr_0, SCALARAVG_vr_1, SCALARAVG_vr_2, SCALARAVG_vr_3, SCALARAVG_vr_4
      real(wp), dimension(num_samples) :: SCALARAVG_vr_5, SCALARAVG_vr_6, SCALARAVG_vr_7, SCALARAVG_vr_8, SCALARAVG_vr_9
      real(wp), dimension(num_samples) :: SCALARAVG_vr_10, SCALARAVG_vr_11, SCALARAVG_vr_12, SCALARAVG_vr_13, SCALARAVG_vr_14
      real(wp), dimension(num_samples) :: Y_SOIL3C, Y_SOIL4C, Y_DEADSTEMC, Y_DEADCROOTC, Y_CWDC, Y_TLAI, Y_GPP

      ! Final input matrix
      real(wp), dimension(num_vars, num_samples), target :: in_data

      ! Output matrix
      real(wp), dimension(num_outputs, output_samples) :: out_data

      ! Layout definitions for tensors
      integer, parameter :: in_layout(2) = [1, 2]
      integer, parameter :: out_layout(2) = [1, 2]

      ! Initialize each variable
      do i = 1, num_samples
         landfrac(i) = real(i, wp) * 0.1_wp
         Latitude(i) = real(i, wp) * 0.2_wp
         Longitude(i) = real(i, wp) * 0.3_wp
         FLDS(i) = real(i, wp) * 0.4_wp
         PSRF(i) = real(i, wp) * 0.5_wp
         FSDS(i) = real(i, wp) * 0.6_wp
         QBOT(i) = real(i, wp) * 0.7_wp
         PRECTmms(i) = real(i, wp) * 0.8_wp
         TBOT(i) = real(i, wp) * 0.9_wp
         LANDFRAC_PFT(i) = real(i, wp) * 1.0_wp
         PCT_NATVEG(i) = real(i, wp) * 1.1_wp
         AREA(i) = real(i, wp) * 1.2_wp
         peatf(i) = real(i, wp) * 1.3_wp
         abm(i) = real(i, wp) * 1.4_wp
         SOIL_COLOR(i) = real(i, wp) * 1.5_wp
         SOIL_ORDER(i) = real(i, wp) * 1.6_wp
         SOIL3C(i) = real(i, wp) * 1.7_wp
         SOIL4C(i) = real(i, wp) * 1.8_wp
         DEADSTEMC(i) = real(i, wp) * 1.9_wp
         DEADCROOTC(i) = real(i, wp) * 2.0_wp
         CWDC(i) = real(i, wp) * 2.1_wp
         TLAI(i) = real(i, wp) * 2.2_wp
         GPP(i) = real(i, wp) * 2.3_wp
         PCT_NAT_PFT_0(i) = real(i, wp) * 2.4_wp
         PCT_NAT_PFT_1(i) = real(i, wp) * 2.5_wp
         PCT_NAT_PFT_2(i) = real(i, wp) * 2.6_wp
         PCT_NAT_PFT_3(i) = real(i, wp) * 2.7_wp
         PCT_NAT_PFT_4(i) = real(i, wp) * 2.8_wp
         PCT_NAT_PFT_5(i) = real(i, wp) * 2.9_wp
         PCT_NAT_PFT_6(i) = real(i, wp) * 3.0_wp
         PCT_NAT_PFT_7(i) = real(i, wp) * 3.1_wp
         PCT_NAT_PFT_8(i) = real(i, wp) * 3.2_wp
         PCT_NAT_PFT_9(i) = real(i, wp) * 3.3_wp
         PCT_NAT_PFT_10(i) = real(i, wp) * 3.4_wp
         PCT_NAT_PFT_11(i) = real(i, wp) * 3.5_wp
         PCT_NAT_PFT_12(i) = real(i, wp) * 3.6_wp
         PCT_NAT_PFT_13(i) = real(i, wp) * 3.7_wp
         PCT_NAT_PFT_14(i) = real(i, wp) * 3.8_wp
         PCT_NAT_PFT_15(i) = real(i, wp) * 3.9_wp
         PCT_NAT_PFT_16(i) = real(i, wp) * 4.0_wp
         PCT_SAND_0(i) = real(i, wp) * 4.1_wp
         PCT_SAND_1(i) = real(i, wp) * 4.2_wp
         PCT_SAND_2(i) = real(i, wp) * 4.3_wp
         PCT_SAND_3(i) = real(i, wp) * 4.4_wp
         PCT_SAND_4(i) = real(i, wp) * 4.5_wp
         PCT_SAND_5(i) = real(i, wp) * 4.6_wp
         PCT_SAND_6(i) = real(i, wp) * 4.7_wp
         PCT_SAND_7(i) = real(i, wp) * 4.8_wp
         PCT_SAND_8(i) = real(i, wp) * 4.9_wp
         PCT_SAND_9(i) = real(i, wp) * 5.0_wp
         SCALARAVG_vr_0(i) = real(i, wp) * 5.1_wp
         SCALARAVG_vr_1(i) = real(i, wp) * 5.2_wp
         SCALARAVG_vr_2(i) = real(i, wp) * 5.3_wp
         SCALARAVG_vr_3(i) = real(i, wp) * 5.4_wp
         SCALARAVG_vr_4(i) = real(i, wp) * 5.5_wp
         SCALARAVG_vr_5(i) = real(i, wp) * 5.6_wp
         SCALARAVG_vr_6(i) = real(i, wp) * 5.7_wp
         SCALARAVG_vr_7(i) = real(i, wp) * 5.8_wp
         SCALARAVG_vr_8(i) = real(i, wp) * 5.9_wp
         SCALARAVG_vr_9(i) = real(i, wp) * 6.0_wp
         SCALARAVG_vr_10(i) = real(i, wp) * 6.1_wp
         SCALARAVG_vr_11(i) = real(i, wp) * 6.2_wp
         SCALARAVG_vr_12(i) = real(i, wp) * 6.3_wp
         SCALARAVG_vr_13(i) = real(i, wp) * 6.4_wp
         SCALARAVG_vr_14(i) = real(i, wp) * 6.5_wp
         Y_SOIL3C(i) = real(i, wp) * 6.6_wp
         Y_SOIL4C(i) = real(i, wp) * 6.7_wp
         Y_DEADSTEMC(i) = real(i, wp) * 6.8_wp
         Y_DEADCROOTC(i) = real(i, wp) * 6.9_wp
         Y_CWDC(i) = real(i, wp) * 7.0_wp
         Y_TLAI(i) = real(i, wp) * 7.1_wp
         Y_GPP(i) = real(i, wp) * 7.2_wp
      end do

      ! Combine individual arrays into the input matrix
      in_data(1, :) = landfrac
      in_data(2, :) = Latitude
      in_data(3, :) = Longitude
      in_data(4, :) = FLDS
      in_data(5, :) = PSRF
      in_data(6, :) = FSDS
      in_data(7, :) = QBOT
      in_data(8, :) = PRECTmms
      in_data(9, :) = TBOT
      in_data(10, :) = LANDFRAC_PFT
      in_data(11, :) = PCT_NATVEG
      in_data(12, :) = AREA
      in_data(13, :) = peatf
      in_data(14, :) = abm
      in_data(15, :) = SOIL_COLOR
      in_data(16, :) = SOIL_ORDER
      in_data(17, :) = SOIL3C
      in_data(18, :) = SOIL4C
      in_data(19, :) = DEADSTEMC
      in_data(20, :) = DEADCROOTC
      in_data(21, :) = CWDC
      in_data(22, :) = TLAI
      in_data(23, :) = GPP
      in_data(24, :) = PCT_NAT_PFT_0
      in_data(25, :) = PCT_NAT_PFT_1
      in_data(26, :) = PCT_NAT_PFT_2
      in_data(27, :) = PCT_NAT_PFT_3
      in_data(28, :) = PCT_NAT_PFT_4
      in_data(29, :) = PCT_NAT_PFT_5
      in_data(30, :) = PCT_NAT_PFT_6
      in_data(31, :) = PCT_NAT_PFT_7
      in_data(32, :) = PCT_NAT_PFT_8
      in_data(33, :) = PCT_NAT_PFT_9
      in_data(34, :) = PCT_NAT_PFT_10
      in_data(35, :) = PCT_NAT_PFT_11
      in_data(36, :) = PCT_NAT_PFT_12
      in_data(37, :) = PCT_NAT_PFT_13
      in_data(38, :) = PCT_NAT_PFT_14
      in_data(39, :) = PCT_NAT_PFT_15
      in_data(40, :) = PCT_NAT_PFT_16
      in_data(41, :) = PCT_SAND_0
      in_data(42, :) = PCT_SAND_1
      in_data(43, :) = PCT_SAND_2
      in_data(44, :) = PCT_SAND_3
      in_data(45, :) = PCT_SAND_4
      in_data(46, :) = PCT_SAND_5
      in_data(47, :) = PCT_SAND_6
      in_data(48, :) = PCT_SAND_7
      in_data(49, :) = PCT_SAND_8
      in_data(50, :) = PCT_SAND_9
      in_data(51, :) = SCALARAVG_vr_0
      in_data(52, :) = SCALARAVG_vr_1
      in_data(53, :) = SCALARAVG_vr_2
      in_data(54, :) = SCALARAVG_vr_3
      in_data(55, :) = SCALARAVG_vr_4
      in_data(56, :) = SCALARAVG_vr_5
      in_data(57, :) = SCALARAVG_vr_6
      in_data(58, :) = SCALARAVG_vr_7
      in_data(59, :) = SCALARAVG_vr_8
      in_data(60, :) = SCALARAVG_vr_9
      in_data(61, :) = SCALARAVG_vr_10
      in_data(62, :) = SCALARAVG_vr_11
      in_data(63, :) = SCALARAVG_vr_12
      in_data(64, :) = SCALARAVG_vr_13
      in_data(65, :) = SCALARAVG_vr_14
      in_data(66, :) = Y_SOIL3C
      in_data(67, :) = Y_SOIL4C
      in_data(68, :) = Y_DEADSTEMC
      in_data(69, :) = Y_DEADCROOTC
      in_data(70, :) = Y_CWDC
      in_data(71, :) = Y_TLAI
      in_data(72, :) = Y_GPP

      ! Load the pre-trained model
      call torch_model_load(model, "LSTM_model.pt")

      ! Map Fortran arrays to PyTorch tensors
      call torch_tensor_from_array(in_tensors(1), in_data, in_layout, torch_kCPU)
      call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)

      ! Perform inference
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Print predicted output
      write(*, '(A)') "Predicted Output (First 5 Rows):"
      do i = 1, 5
         write(*, '(7F10.4)') out_data(:, i)  ! Print the first 5 rows of output data
      end do

      ! Clean up resources
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)

   end subroutine main

end program offline_training_inference
