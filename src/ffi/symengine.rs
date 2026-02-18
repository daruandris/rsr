use std::ffi::{CStr, CString};
use std::os::raw::c_char;

unsafe extern "C" {
    fn simplify_symengine_cpp(input: *const c_char, output: *mut c_char, max_len: usize);
}

pub fn simplify_symengine(eq: &str) -> String {
    let c_input = CString::new(eq).unwrap_or_default();
    const BUFFER_SIZE: usize = 4096;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    unsafe {
        simplify_symengine_cpp(
            c_input.as_ptr(), 
            buffer.as_mut_ptr() as *mut c_char, 
            BUFFER_SIZE
        );
        CStr::from_ptr(buffer.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    }
}